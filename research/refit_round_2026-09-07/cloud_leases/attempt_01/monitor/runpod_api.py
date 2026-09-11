"""Secret-safe RunPod client for the Elastic Reasoner continuation (runpod_v1).

The API key is read from the key file named by RUNPOD_KEY_FILE (default
~/Documents/Credentials/fidelio.txt) inside this process only, sent solely as an
Authorization header to official RunPod endpoints over HTTPS, and never printed,
logged, hashed for display, placed in a URL/argument, or forwarded to a Pod.
Line 1 of the key file is the read-only key; line 3 is the write-capable key (the
established local convention); reads use line 1, mutations use line 3.

Surfaces (verified against the current docs / OpenAPI on 2026-09-06):
  GraphQL  https://api.runpod.io/graphql   myself{clientBalance currentSpendPerHr pods{...}},
           podFindAndDeployOnDemand(input: {... terminateAfter}) (provider-enforced TTL), podTerminate
  REST v2  https://api.runpod.io/v2        /catalog/gpus (price, availability), /pods, /pods/{id},
           /pods/{id}/action {stop|terminate}, /billing, /billing/pods, /account/ssh-keys
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

GQL = "https://api.runpod.io/graphql"
V2 = "https://api.runpod.io/v2"
UA = "elastic-reasoner-runpod_v1"
KEY_FILE = Path(os.environ.get("RUNPOD_KEY_FILE", str(Path.home() / "Documents/Credentials/fidelio.txt")))
_KEY_RE = re.compile(r"^rpa_[A-Za-z0-9]{20,}$")


class APIError(RuntimeError):
    def __init__(self, msg: str, status: int | None = None):
        super().__init__(msg)
        self.status = status  # HTTP status when the provider answered; None for network/other failures


def _key(write: bool) -> str:
    lines = [l.strip() for l in KEY_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lines:
        raise APIError("key file empty")
    k = lines[2] if (write and len(lines) >= 3) else lines[0]
    if not _KEY_RE.match(k):
        raise APIError("key file line does not look like a RunPod API key")
    return k


def _redact(text: str) -> str:
    return re.sub(r"rpa_[A-Za-z0-9]{10,}", "rpa_[REDACTED]", text)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise APIError("RunPod API redirect refused")


def _request(method: str, url: str, body=None, write: bool = False, timeout: int = 60):
    origin = urllib.parse.urlsplit(url)
    if (origin.scheme != "https" or origin.netloc != "api.runpod.io" or origin.fragment
            or not (origin.path == "/graphql" or origin.path == "/v2" or origin.path.startswith("/v2/"))):
        raise APIError("refusing to send credentials outside official RunPod API endpoints")
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, method=method, data=data, headers={
        "Authorization": "Bearer " + _key(write), "Content-Type": "application/json", "User-Agent": UA})
    try:
        with urllib.request.build_opener(_NoRedirect()).open(req, timeout=timeout) as r:
            raw = r.read().decode()
            return r.status, (json.loads(raw) if raw.strip() else {})
    except urllib.error.HTTPError as e:
        raise APIError(f"HTTP {e.code} {method} {url.split('?')[0]}: {_redact(e.read().decode()[:300])}", status=int(e.code)) from None
    except urllib.error.URLError as e:
        raise APIError(f"network error {method} {url.split('?')[0]}: {_redact(str(e))}") from None


def gql(query: str, variables=None, write: bool = False) -> dict:
    st, body = _request("POST", GQL, {"query": query, "variables": variables or {}}, write=write)
    if not isinstance(body, dict) or "errors" in body or not isinstance(body.get("data"), dict):
        raise APIError("GraphQL error: " + _redact(json.dumps(body)[:400]))
    return body["data"]


def v2(method: str, path: str, body=None, write: bool = False):
    st, out = _request(method, V2 + path, body, write=write)
    return out


# ------------------------------------------------------------------ read-only account views
def balance() -> dict:
    d = gql("{ myself { clientBalance currentSpendPerHr } }")["myself"]
    return {"clientBalance": float(d["clientBalance"]), "currentSpendPerHr": float(d.get("currentSpendPerHr") or 0)}


def my_pods() -> list[dict]:
    d = gql("{ myself { pods { id name desiredStatus costPerHr machineId runtime { uptimeInSeconds } } } }")
    return list(d["myself"].get("pods") or [])


def pods_v2() -> list[dict]:
    out = v2("GET", "/pods")
    return out if isinstance(out, list) else out.get("pods", out.get("data", []))


def pod_v2(pod_id: str) -> dict:
    return v2("GET", f"/pods/{pod_id}")


def gpu_catalog(gpu_ids: list[str] | None = None) -> list[dict]:
    out = v2("GET", "/catalog/gpus")
    gpus = out.get("gpus", out) if isinstance(out, dict) else out
    if gpu_ids:
        gpus = [g for g in gpus if g.get("id") in gpu_ids or any(s.lower() in (g.get("name") or "").lower() for s in gpu_ids)]
    return gpus


def gpu_offer(gpu_type_id: str) -> dict:
    """Live secure-cloud on-demand price for one GPU type via GraphQL (the surface the run-9 controller used)."""
    q = """query Offer($id: String!) { gpuTypes(input: {id: $id}) { id displayName memoryInGb secureCloud
             lowestPrice(input: {gpuCount: 1, secureCloud: true}) { uninterruptablePrice stockStatus } } }"""
    d = gql(q, {"id": gpu_type_id})
    return d["gpuTypes"][0] if d.get("gpuTypes") else {}


def billing(last_n: int = 7) -> dict:
    return v2("GET", f"/billing?lastN={int(last_n)}")


def billing_pods(pod_id: str | None = None, last_n: int = 7) -> dict:
    q = f"/billing/pods?lastN={int(last_n)}" + (f"&podId={pod_id}" if pod_id else "")
    return v2("GET", q)


def ssh_keys() -> list[str]:
    out = v2("GET", "/account/ssh-keys")
    return list(out.get("keys", []))


# ------------------------------------------------------------------ mutations (write key)
def deploy(name: str, gpu_type_id: str, image: str, disk_gb: int, terminate_after_iso: str, env: dict, ports: str = "22/tcp",
           min_vcpu: int = 8, min_mem_gb: int = 32, cloud: str = "SECURE") -> dict:
    """podFindAndDeployOnDemand with a provider-enforced terminateAfter (ISO8601 Z)."""
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", terminate_after_iso):
        raise APIError("terminateAfter must be an exact ISO8601 UTC timestamp")
    q = """mutation Deploy($input: PodFindAndDeployOnDemandInput!) {
      podFindAndDeployOnDemand(input: $input) { id machineId costPerHr desiredStatus name machine { gpuDisplayName } } }"""
    variables = {"input": {"cloudType": cloud, "gpuCount": 1, "gpuTypeId": gpu_type_id, "name": name, "imageName": image,
                           "containerDiskInGb": int(disk_gb), "volumeInGb": 0, "minVcpuCount": int(min_vcpu), "minMemoryInGb": int(min_mem_gb),
                           "ports": ports, "terminateAfter": terminate_after_iso,
                           "env": [{"key": k, "value": v} for k, v in env.items()]}}
    d = gql(q, variables, write=True)
    pod = d.get("podFindAndDeployOnDemand")
    if not isinstance(pod, dict) or not pod.get("id"):
        raise APIError("deploy response carried no pod id")
    return pod


def terminate(pod_id: str) -> bool:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", pod_id):
        raise APIError("bad pod id")
    d = gql(f'mutation {{ podTerminate(input: {{podId: {json.dumps(pod_id)}}}) }}', write=True)
    # podTerminate returns Void: a null value without GraphQL errors (gql() raises on errors) is the success response
    # (observed 2026-09-06 on pod 12t3dvdxr2c1l3: {"podTerminate": null} and the pod was gone immediately after)
    return "podTerminate" in d and d["podTerminate"] is not False


def terminate_v2(pod_id: str) -> bool:
    st, _ = _request("DELETE", f"{V2}/pods/{pod_id}", write=True)
    return st in (200, 204)


def pod_gone(pod_id: str) -> bool:
    """True only when BOTH surfaces positively report absence: GraphQL does not list the pod as live and
    REST v2 answers exactly HTTP 404 (or a terminated status).  Any other failure raises (uncertain)."""
    live_gql = {p.get("id") for p in my_pods() if str(p.get("desiredStatus", "")).upper() not in ("TERMINATED", "EXITED")}
    try:
        p = pod_v2(pod_id)
        live_v2 = str(p.get("status", p.get("desiredStatus", ""))).upper() not in ("TERMINATED", "EXITED")
    except APIError as e:
        if e.status == 404:
            live_v2 = False
        else:
            raise
    return (pod_id not in live_gql) and (not live_v2)


if __name__ == "__main__":  # read-only self-check; prints parsed fields only
    b = balance(); print("balance", b)
    print("live pods (gql):", [(p["id"], p.get("name"), p.get("desiredStatus"), p.get("costPerHr")) for p in my_pods()])
    for g in gpu_catalog(["A100"]):
        print("catalog", g.get("id"), g.get("name"), "mem", g.get("memory"), "secure", g.get("secure"), "price", g.get("price"), "avail", g.get("availability"), "cuda", [c.get("version") for c in (g.get("cudaVersions") or []) if c.get("available")])
    print("ssh keys registered:", len(ssh_keys()))
