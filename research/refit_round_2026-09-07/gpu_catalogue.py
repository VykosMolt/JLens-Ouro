"""Read public RunPod GPU prices and stock; never query account or create Pods."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import urllib.request


QUERY = """query FollowupGpuCatalogue {
  gpuTypes {
    id displayName memoryInGb secureCloud communityCloud
    secure: lowestPrice(input: {gpuCount: 1, secureCloud: true}) {
      stockStatus uninterruptablePrice availableGpuCounts
      minMemory minVcpu minDownload minUpload supportPublicIp
    }
    community: lowestPrice(input: {gpuCount: 1, secureCloud: false}) {
      stockStatus uninterruptablePrice availableGpuCounts
      minMemory minVcpu minDownload minUpload supportPublicIp
    }
  }
}"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("gpu_catalogue.json"))
    args = parser.parse_args()
    request = urllib.request.Request(
        "https://api.runpod.io/graphql",
        json.dumps({"query": QUERY}).encode(),
        {"Content-Type": "application/json", "User-Agent": "curl/8.0"},
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        envelope = json.load(response)
    if envelope.get("errors"):
        raise RuntimeError(json.dumps(envelope["errors"]))
    data = envelope["data"]
    result = {
        "retrieved_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "endpoint": "https://api.runpod.io/graphql",
        "purpose": "Read-only one-GPU on-demand price and stock comparison; no account query or mutation.",
        "query": QUERY,
        "data": data,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for gpu in sorted(data["gpuTypes"], key=lambda item: (-item["memoryInGb"], item["id"])):
        if gpu["memoryInGb"] < 24:
            continue
        offers = {
            cloud: gpu[cloud]
            for cloud in ("secure", "community")
            if gpu.get(cloud) is not None
        }
        print(json.dumps({"id": gpu["id"], "vram_gb": gpu["memoryInGb"], **offers}))


if __name__ == "__main__":
    main()
