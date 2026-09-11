"""Reanalyse retained Ouro outputs without modifying the submitted artifacts.

Run with ouro_project/venv/bin/python. All ranks in source arrays are zero-based.
No original metric/report helpers are imported. See RESEARCH_LOG.md for the
analysis choices made before calculating the follow-up layer curves.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ARTIFACTS = Path("/home/moloch/ouro_project/artifacts/jlens")
MAIN = ARTIFACTS / "retrieved/jlens-b300-20260905-0359/eval/n100_exit3"
LOCAL = ARTIFACTS / "eval/b300_local_allexits_strict"
POS2 = ARTIFACTS / "eval/b300_local_allexits_pos-2_strict"
METHODS = ("jlens_exit3", "logitlens")
OPERATIONS = {"addition", "subtraction", "multiplication", "division", "mod", "squared"}
NUMBER_WORDS = dict(zip("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split(), range(21)))
NUMBER_WORDS["third"] = 3
TASKS = ("multihop", "order-ops")
TASK_LABELS = {"multihop": "Multihop", "order-ops": "Arithmetic"}
METHOD_LABELS = {"jlens_exit3": "J-Lens", "logitlens": "Logit lens"}
COLORS = {"jlens_exit3": "#bc4637", "logitlens": "#27699b"}
N_BOOT = 20_000


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def save_json(path, data):
    def convert(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(type(x))
    Path(path).write_text(json.dumps(data, indent=2, default=convert, allow_nan=False) + "\n")


def save_csv(path, rows):
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def load_run(path):
    items = json.loads((path / "items.json").read_text())
    names = json.loads((path / "task_names.json").read_text())
    with np.load(path / "arrays.npz", allow_pickle=False) as a:
        arrays = {k: a[k] for k in a.files}
    assert len(items) == 148 and len({i["name"] for i in items}) == 148
    assert np.array_equal(arrays["exit_top1"], arrays["logitlens_top1"][:, [47, 95, 143, 191]])
    for method in METHODS:
        r = arrays[method + "_allrank"]
        assert r.shape == (148, 128, 192)
        for i, item in enumerate(items):
            n = len(names[item["task"]])
            assert np.all(r[i, :n] >= 0) and np.all(r[i, n:] == -1)
            for s, own in enumerate(item["own_index"]):
                if own >= 0:
                    assert names[item["task"]][own] == item["intermediates"][s]
                    assert np.array_equal(r[i, own], arrays[method + "_rank"][i, s])
    return items, names, arrays


def eligible_slots(item):
    return [s for s, name in enumerate(item["intermediates"])
            if item["scorable"][s] and not item["leaked"][s]
            and (item["task"] == "multihop" or name not in OPERATIONS)]


def canonical(name):
    return str(NUMBER_WORDS.get(name.lower(), int(name) if name.isdigit() else name.lower()))


def concept_clusters(items, indices):
    """Connect items sharing any eligible concept, including digit/word aliases.

    This is a sensitivity analysis of dependence, not an asserted iid sampling
    scheme for the synthetic task set. Item weights are retained by ratio means.
    """
    parents = list(range(len(indices)))
    def find(a):
        while parents[a] != a:
            parents[a] = parents[parents[a]]
            a = parents[a]
        return a
    first = {}
    for j, i in enumerate(indices):
        for s in eligible_slots(items[i]):
            label = canonical(items[i]["intermediates"][s])
            if label in first:
                parents[find(j)] = find(first[label])
            else:
                first[label] = j
    return np.array([find(j) for j in range(len(indices))])


def bootstrap_weights(n, seed=0, clusters=None):
    rng = np.random.default_rng(seed)
    if clusters is None:
        selected = rng.integers(0, n, (N_BOOT, n))
        weights = np.zeros((N_BOOT, n), dtype=np.float64)
        np.add.at(weights, (np.arange(N_BOOT)[:, None], selected), 1 / n)
        return weights
    _, group = np.unique(clusters, return_inverse=True)
    g = group.max() + 1
    selected = rng.integers(0, g, (N_BOOT, g))
    counts = np.zeros((N_BOOT, g), dtype=np.float64)
    np.add.at(counts, (np.arange(N_BOOT)[:, None], selected), 1)
    weights = counts[:, group]
    return weights / weights.sum(axis=1, keepdims=True)


def boot_means(values, weights):
    return (weights @ values.reshape(len(values), -1)).reshape(N_BOOT, *values.shape[1:])


def interval(draws):
    return np.quantile(draws, [0.025, 0.975], axis=0)


def summary(values, weights):
    return {"mean": values.mean(axis=0), "ci95": np.moveaxis(interval(boot_means(values, weights)), 0, -1)}


def score_items(items, names, arrays, task, *, canonical_controls=False):
    indices = np.array([i for i, item in enumerate(items) if item["task"] == task and eligible_slots(item)])
    output = {"indices": indices, "n_slots": sum(len(eligible_slots(items[i])) for i in indices)}
    for method in METHODS:
        ranks = arrays[method + "_allrank"].reshape(148, 128, 4, 48)
        rows = {k: [] for k in ("hit", "control", "excess", "any_hit", "any_control", "any_excess")}
        for i in indices:
            item = items[i]
            own = {j for j in item["own_index"] if j >= 0}
            own_concepts = {canonical(names[task][j]) for j in own}
            slot_rows = {k: [] for k in rows}
            for s in eligible_slots(item):
                j = item["own_index"][s]
                ctrl = [c for c, nm in enumerate(names[task]) if c not in own
                        and ((nm in OPERATIONS) == (names[task][j] in OPERATIONS))]
                if canonical_controls:
                    # Optional sensitivity: one entry per concept, exclude
                    # equivalent own concepts. Rank of a concept is min over
                    # its names/forms; default historical controls unchanged.
                    groups = {}
                    for c in ctrl:
                        key = canonical(names[task][c])
                        if key not in own_concepts:
                            groups.setdefault(key, []).append(c)
                    control_ranks = np.stack([ranks[i, cs].min(axis=0) for cs in groups.values()])
                else:
                    control_ranks = ranks[i, ctrl]
                assert len(control_ranks) > 0
                own_hit = ranks[i, j] < 10
                control_hit = control_ranks < 10
                hit = own_hit.astype(float)
                control = control_hit.mean(axis=0)
                any_hit = own_hit.any(axis=-1).astype(float)
                any_control = control_hit.any(axis=-1).mean(axis=0)
                for k, v in dict(hit=hit, control=control, excess=hit-control,
                                 any_hit=any_hit, any_control=any_control, any_excess=any_hit-any_control).items():
                    slot_rows[k].append(v)
            for k in rows:
                rows[k].append(np.mean(slot_rows[k], axis=0))
        output[method] = {k: np.stack(v) for k, v in rows.items()}
    return output


def plot_finish(fig, out, name):
    fig.savefig(out / (name + ".png"), dpi=180, bbox_inches="tight")
    fig.savefig(out / (name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def layer_analysis(items, names, arrays, out):
    scores = {task: score_items(items, names, arrays, task) for task in TASKS}
    results, rows, bands, saved = {}, [], [], {}
    maxima = np.zeros(N_BOOT)
    cluster_maxima = np.zeros(N_BOOT)
    for task_index, task in enumerate(TASKS):
        s = scores[task]
        indices = s["indices"]
        clusters = concept_clusters(items, indices)
        w = bootstrap_weights(len(indices), seed=task_index)
        wc = bootstrap_weights(len(indices), seed=task_index, clusters=clusters)
        delta = s[METHODS[0]]["excess"] - s[METHODS[1]]["excess"]
        draws = boot_means(delta, w)
        cluster_draws = boot_means(delta, wc)
        maxima = np.maximum(maxima, abs(draws-delta.mean(axis=0)).max(axis=(1, 2)))
        cluster_maxima = np.maximum(cluster_maxima, abs(cluster_draws-delta.mean(axis=0)).max(axis=(1, 2)))
        result = {"n_items": len(indices), "n_slots": s["n_slots"], "n_concept_clusters": len(np.unique(clusters)),
                  "item_indices": indices, "concept_clusters": clusters,
                  "methods": {}, "difference": summary(delta, w),
                  "difference_cluster": summary(delta, wc),
                  "any_layer_difference": summary(s[METHODS[0]]["any_excess"] - s[METHODS[1]]["any_excess"], bootstrap_weights(len(indices)))}
        for method in METHODS:
            result["methods"][method] = {k: summary(v, w) for k, v in s[method].items()}
            for k, v in s[method].items():
                saved[f"{task}_{method}_{k}"] = v
        saved[task + "_indices"] = indices
        saved[task + "_concept_clusters"] = clusters
        # Slope per full physical stack (depth scaled to 0..1).
        x = np.linspace(0, 1, 48)
        x -= x.mean()
        slope = np.einsum("iul,l->iu", delta, x) / (x @ x)
        result["depth_slope"] = summary(slope, w)
        result["depth_slope_cluster"] = summary(slope, wc)
        result["depth_bands"] = {}
        for start, stop in ((0,16),(16,32),(32,48)):
            band = delta[:, :, start:stop].mean(axis=-1)
            name = f"{start+1}-{stop}"
            result["depth_bands"][name] = {"difference": summary(band, w), "difference_cluster": summary(band, wc)}
            for u in range(4):
                entry = {"task": task, "loop": u+1, "layers": name, "n_items": len(indices)}
                for label, ww in (("item",w),("cluster",wc)):
                    z = summary(band, ww)
                    entry[label+"_difference"] = z["mean"][u]
                    entry[label+"_ci_low"], entry[label+"_ci_high"] = z["ci95"][u]
                bands.append(entry)
        results[task] = result
    radius = float(np.quantile(maxima, 0.95))
    cluster_radius = float(np.quantile(cluster_maxima, 0.95))
    for task in TASKS:
        r = results[task]
        mean = r["difference"]["mean"]
        r["difference"]["simultaneous_ci95_384"] = np.stack([mean-radius, mean+radius], axis=-1)
        r["difference_cluster"]["simultaneous_ci95_384"] = np.stack([mean-cluster_radius, mean+cluster_radius], axis=-1)
        for u in range(4):
            for l in range(48):
                row = {"task": task, "loop": u+1, "physical_layer": l+1, "virtual_index_zero_based": u*48+l,
                       "n_items": r["n_items"], "n_slots": r["n_slots"], "n_concept_clusters": r["n_concept_clusters"]}
                for method in METHODS:
                    for metric in ("hit", "control", "excess"):
                        stat = r["methods"][method][metric]
                        row[f"{method}_{metric}"] = stat["mean"][u,l]
                        row[f"{method}_{metric}_ci_low"],row[f"{method}_{metric}_ci_high"] = stat["ci95"][u,l]
                row["difference"] = mean[u,l]
                row["ci_low"],row["ci_high"] = r["difference"]["ci95"][u,l]
                row["sim_low"],row["sim_high"] = r["difference"]["simultaneous_ci95_384"][u,l]
                row["cluster_ci_low"],row["cluster_ci_high"] = r["difference_cluster"]["ci95"][u,l]
                row["cluster_sim_low"],row["cluster_sim_high"] = r["difference_cluster"]["simultaneous_ci95_384"][u,l]
                rows.append(row)
    save_csv(out / "layerwise.csv", rows)
    save_csv(out / "depth_bands.csv", bands)
    np.savez_compressed(out / "item_scores.npz", **saved)
    save_json(out / "layerwise.json", {"bootstrap": {"draws": N_BOOT, "historical_seed": 0, "layer_task_seeds": [0,1], "simultaneous_cells":384,
              "simultaneous_absolute_radius":radius, "cluster_simultaneous_absolute_radius":cluster_radius}, "tasks":results})
    layer_plots(results, out)
    return scores, results


def layer_plots(results, out):
    x = np.arange(1,49)
    for kind in ("performance", "difference"):
        fig, axes = plt.subplots(2,4,figsize=(15,7),sharex=True,sharey=True,layout="constrained")
        for ti, task in enumerate(TASKS):
            r = results[task]
            for u in range(4):
                ax = axes[ti,u]
                ax.axhline(0,color="#555555",lw=.7)
                if kind == "performance":
                    for method in METHODS:
                        st = r["methods"][method]["excess"]
                        ax.plot(x,st["mean"][u],color=COLORS[method],label=METHOD_LABELS[method],lw=1.6)
                        ci = st["ci95"][u]
                        ax.fill_between(x,ci[:,0],ci[:,1],color=COLORS[method],alpha=.15)
                else:
                    st = r["difference"]
                    ci = st["simultaneous_ci95_384"][u]
                    ax.fill_between(x,ci[:,0],ci[:,1],color="#aaa6a1",alpha=.25,label="95% simultaneous, 384 cells")
                    ci = st["ci95"][u]
                    ax.fill_between(x,ci[:,0],ci[:,1],color="#6c486f",alpha=.25,label="95% pointwise")
                    ax.plot(x,st["mean"][u],color="#623d68",lw=1.6,label="J-Lens − logit lens")
                ax.set_title(f"{TASK_LABELS[task]} · loop {u+1} · n={r['n_items']}")
                ax.set_xticks([1,16,32,48])
                if ti==1: ax.set_xlabel("Physical layer (1–48)")
                if u==0: ax.set_ylabel("Excess hit@10" if kind=="performance" else "Paired difference in excess hit@10")
        axes[0,0].legend(fontsize=8,loc="upper left")
        fig.suptitle("Ouro · 100-prompt final-exit J-Lens · last prompt token\nPaired item bootstrap; uncertainty conditional on one fit",fontsize=13)
        plot_finish(fig,out,"layerwise_"+kind)
    fig,axes=plt.subplots(2,3,figsize=(14,6),layout="constrained")
    for ti,task in enumerate(TASKS):
        r=results[task]
        fields=[r["methods"][m]["excess"]["mean"] for m in METHODS]+[r["difference"]["mean"]]
        for j,(data,title) in enumerate(zip(fields,["J-Lens","Logit lens","J-Lens − logit lens"])):
            ax=axes[ti,j]
            im=ax.imshow(data,aspect="auto",origin="upper",extent=[.5,48.5,4.5,.5],cmap="RdBu_r",vmin=-.6,vmax=.6)
            ax.set_title(TASK_LABELS[task]+" · "+title)
            ax.set_yticks([1,2,3,4]);ax.set_xticks([1,16,32,48]);ax.set_xlabel("Physical layer")
            if j==0:ax.set_ylabel("Recurrent loop")
    fig.colorbar(im,ax=axes,label="Excess hit@10 / paired difference",shrink=.8)
    plot_finish(fig,out,"loop_layer_heatmap")


def exit_analysis(out):
    all_results, matrix_rows, diagnostic_rows, actual_rows = {}, [], [], []
    windows = {"1-47": slice(0,47), "41-47": slice(40,47), "41-48": slice(40,48), "48": slice(47,48)}
    lens_labels = ["Local J-Lens", "Final J-Lens", "Logit lens", "Actual loop exit", "Actual final exit"]
    for position, path in (("-1",LOCAL),("-2",POS2)):
        items, names, a = load_run(path)
        n = len(items)
        w = bootstrap_weights(n, seed=20260907)
        ground = a["exit_top1"]
        actual = (ground[:,:,None] == ground[:,None,:]).astype(float)
        actual_stats = summary(actual, w)
        for u in range(4):
            for v in range(4):
                actual_rows.append({"position":position,"exit_a":u+1,"exit_b":v+1,"n_items":n,
                                    "agreement":actual_stats["mean"][u,v],
                                    "ci_low":actual_stats["ci95"][u,v,0],"ci_high":actual_stats["ci95"][u,v,1]})
        res={"path":path,"n_items":n,"actual_exit_agreement":actual_stats,"loops":{}}
        for u in range(4):
            ix = slice(u*48,(u+1)*48)
            prefixes=[f"jlens_exit{u}","jlens_exit3","logitlens"]
            top=[a[p+"_top1"][:,ix] for p in prefixes]
            top += [np.broadcast_to(ground[:,u,None],(n,48)),np.broadcast_to(ground[:,-1,None],(n,48))]
            top=np.stack(top,axis=-1)  # item, layer, estimator
            agreement=(top[:,:,:,None] == top[:,:,None,:]).astype(float)
            r={"matrix_labels":lens_labels,"windows":{}}
            for window, sl in windows.items():
                matrix=summary(agreement[:,sl].mean(axis=1),w)
                r["windows"][window]={"top1_agreement_matrix":matrix,"diagnostics":[]}
                for j,label in enumerate(lens_labels):
                    for k,other in enumerate(lens_labels):
                        matrix_rows.append({"position":position,"loop":u+1,"layers":window,"estimator_a":label,"estimator_b":other,
                                            "n_items":n,"n_layers":top[:,sl].shape[1],"agreement":matrix["mean"][j,k],
                                            "ci_low":matrix["ci95"][j,k,0],"ci_high":matrix["ci95"][j,k,1]})
                for j,prefix in enumerate(prefixes):
                    for target,label in (("local","actual_current_exit"),("final","actual_final_exit")):
                        # final-J and LL `local` fields are FINAL exit. Only the
                        # local-exit lens retains current-exit ranks/KL at u<3.
                        retained = target == "final" or j == 0 or u == 3
                        record={"position":position,"loop":u+1,"layers":window,"lens":lens_labels[j],"target":label,
                                "n_items":n,"rank_kl_retained":retained,
                                "top1_agreement":matrix["mean"][j,3 if target=="local" else 4],
                                "top1_ci_low":matrix["ci95"][j,3 if target=="local" else 4,0],
                                "top1_ci_high":matrix["ci95"][j,3 if target=="local" else 4,1]}
                        metrics={}
                        if retained:
                            ranks=a[prefix+"_rank_of_"+target+"_top1"][:,ix][:,sl]
                            assert np.all(ranks >= 0)
                            metrics={"actual_token_hit10":(ranks<10).mean(axis=1),
                                     "actual_token_hit100":(ranks<100).mean(axis=1),
                                     "actual_token_mrr":(1/(ranks+1)).mean(axis=1),
                                     "actual_token_mean_log10_rank":np.log10(ranks+1).mean(axis=1),
                                     "kl_lens_to_actual":a[prefix+"_kl_to_"+target][:,ix][:,sl].mean(axis=1)}
                        for key in ("actual_token_hit10","actual_token_hit100","actual_token_mrr","actual_token_mean_log10_rank","kl_lens_to_actual"):
                            if key in metrics:
                                st=summary(metrics[key],w)
                                record[key]=float(st["mean"])
                                record[key+"_ci_low"],record[key+"_ci_high"]=st["ci95"]
                            else:
                                record[key]=record[key+"_ci_low"]=record[key+"_ci_high"]=None
                        diagnostic_rows.append(record)
                        r["windows"][window]["diagnostics"].append(record)
            if u<3:
                r["kl_local_lens_to_final_lens_all48"]=summary(a[f"jlens_exit{u}_kl_to_eventual_readout"][:,ix].mean(axis=1),w)
            r["actual_exit_to_final_kl"]=summary(a["logitlens_kl_to_final"][:,u*48+47],w)
            r["per_layer_to_current_exit"]={label:summary((top[:,:,j]==ground[:,u,None]).astype(float),w) for j,label in enumerate(lens_labels[:3])}
            res["loops"][str(u+1)]=r
        all_results[position]=res
    save_json(out/"exit_agreement.json",all_results)
    save_csv(out/"exit_agreement_matrix.csv",matrix_rows)
    save_csv(out/"exit_diagnostics.csv",diagnostic_rows)
    save_csv(out/"actual_exit_agreement.csv",actual_rows)
    fig,axes=plt.subplots(1,4,figsize=(16,4.7),layout="constrained")
    short=["Local J","Final J","Logit","Exit k","Final"]
    for u,ax in enumerate(axes):
        data=np.array(all_results["-1"]["loops"][str(u+1)]["windows"]["41-47"]["top1_agreement_matrix"]["mean"])
        im=ax.imshow(data,vmin=0,vmax=1,cmap="Blues")
        for j in range(5):
            for k in range(5):
                ax.text(k,j,f"{data[j,k]*100:.1f}",ha="center",va="center",fontsize=8,color="white" if data[j,k]>.6 else "black")
        ax.set_xticks(range(5),short,rotation=45,ha="right");ax.set_yticks(range(5),short)
        ax.set_title(f"Loop {u+1}")
    fig.colorbar(im,ax=axes,label="Top-1 agreement",shrink=.65)
    fig.suptitle("Ouro · agreement with actual exits · 148 prompts\nPhysical layers 41–47; guaranteed identity boundary excluded; cells in %")
    plot_finish(fig,out,"exit_agreement_matrix")
    fig,axes=plt.subplots(1,2,figsize=(9,4.2),layout="constrained")
    for ax,pos in zip(axes,["-1","-2"]):
        data=all_results[pos]["actual_exit_agreement"]["mean"]
        im=ax.imshow(data,vmin=0,vmax=1,cmap="Blues")
        for u in range(4):
            for v in range(4):
                ax.text(v,u,f"{100*data[u,v]:.1f}",ha="center",va="center",color="white" if data[u,v]>.6 else "black")
        ax.set_xticks(range(4),range(1,5));ax.set_yticks(range(4),range(1,5));ax.set_xlabel("Exit loop");ax.set_ylabel("Exit loop")
        ax.set_title(f"Prompt position {pos} · n=148")
    fig.colorbar(im,ax=axes,label="Top-1 agreement",shrink=.8)
    fig.suptitle("Actual Ouro exits · no fitted lens · cells in %")
    plot_finish(fig,out,"actual_exit_agreement")
    fig,axes=plt.subplots(1,4,figsize=(15,3.8),sharey=True,layout="constrained")
    for u,ax in enumerate(axes):
        for label,color in zip(lens_labels[:3],["#328a69",COLORS["jlens_exit3"],COLORS["logitlens"]]):
            st=all_results["-1"]["loops"][str(u+1)]["per_layer_to_current_exit"][label]
            ax.plot(np.arange(1,49),st["mean"],label=label,color=color)
        ax.set_title(f"Loop {u+1}");ax.set_xlabel("Physical layer");ax.set_xticks([1,16,32,48]);ax.axvline(48,color="#555",ls=":",lw=.8)
    axes[0].set_ylabel("Agreement with actual current exit");axes[0].legend(fontsize=8)
    fig.suptitle("Ouro · agreement by physical layer · last prompt token\nLayer 48 is the native exit (local J-Lens and logit lens agree there by construction)")
    plot_finish(fig,out,"exit_agreement_by_layer")


def criterion_pass(item, criterion):
    answer=item["continuation"].strip().strip('"').lower()
    target=item["target"].strip().lower()
    def exact_prefix(t):
        return answer.startswith(t) and (len(answer)==len(t) or not answer[len(t)].isalnum())
    def numeric_prefix(t):
        if not exact_prefix(t): return False
        return not (re.fullmatch(r"[+-]?\d+",t) and re.match(r"(?:[.,/]\d|[eE][+-]?\d)",answer[len(t):]))
    if criterion=="historical": return exact_prefix(target)
    if criterion=="strict_numeric": return numeric_prefix(target)
    if criterion=="numeric_equivalence":
        return numeric_prefix(target) or (target in NUMBER_WORDS and target!="third" and numeric_prefix(str(NUMBER_WORDS[target])))
    raise ValueError(criterion)


def correctness_analysis(items,names,arrays,out):
    criteria=("historical","strict_numeric","numeric_equivalence")
    pass_masks={c:np.array([criterion_pass(it,c) for it in items],dtype=bool) for c in criteria}
    assert [int(pass_masks[c].sum()) for c in criteria]==[72,70,71]
    scores={task:score_items(items,names,arrays,task) for task in TASKS}
    result={"all_148_counts":{c:{"passing":int(mask.sum()),"failing":int((~mask).sum())} for c,mask in pass_masks.items()},"criteria":{}}
    rows=[]
    for c in criteria:
        result["criteria"][c]={}
        for task in TASKS:
            s=scores[task];idx=s["indices"];mask=pass_masks[c][idx]
            cluster=concept_clusters(items,idx)
            wc=bootstrap_weights(len(idx),seed=19,clusters=cluster)
            r={"n_passing":int(mask.sum()),"n_failing":int((~mask).sum()),"strata":{}}
            stratum_weights={}
            for label,keep,seed in (("passing",mask,0),("failing",~mask,1)):
                w=bootstrap_weights(int(keep.sum()),seed=seed)
                wsub=wc[:,keep]
                assert np.all(wsub.sum(axis=1)>0), "A bootstrap draw loses a whole stratum"
                wsub=wsub/wsub.sum(axis=1,keepdims=True)
                stratum_weights[label]=(w,wsub)
                stat={"n_items":int(keep.sum()),"methods":{}}
                for method in METHODS:
                    stat["methods"][method]={key:summary(values[keep],w) for key,values in s[method].items()}
                for key in ("excess","any_excess"):
                    d=s[METHODS[0]][key][keep]-s[METHODS[1]][key][keep]
                    stat[key+"_difference"]=summary(d,w)
                    stat[key+"_difference_cluster"]=summary(d,wsub)
                r["strata"][label]=stat
                for u in range(4):
                    row={"criterion":c,"task":task,"stratum":label,"n_items":int(keep.sum()),"loop":u+1}
                    for method in METHODS:
                        for metric in ("any_hit","any_control","any_excess"):
                            z=stat["methods"][method][metric]
                            row[method+"_"+metric]=z["mean"][u]
                            row[method+"_"+metric+"_ci_low"],row[method+"_"+metric+"_ci_high"]=z["ci95"][u]
                    z=stat["any_excess_difference"]
                    row["difference"]=z["mean"][u];row["ci_low"],row["ci_high"]=z["ci95"][u]
                    z=stat["any_excess_difference_cluster"]
                    row["cluster_ci_low"],row["cluster_ci_high"]=z["ci95"][u]
                    rows.append(row)
            # Difference of paired method effects, passing minus failing.
            # Conditional cluster weights couple shared concepts across strata.
            d=s[METHODS[0]]["any_excess"]-s[METHODS[1]]["any_excess"]
            r["interaction_passing_minus_failing"]={"mean":d[mask].mean(axis=0)-d[~mask].mean(axis=0)}
            for label,w_index in (("item",0),("cluster",1)):
                draws=boot_means(d[mask],stratum_weights["passing"][w_index])-boot_means(d[~mask],stratum_weights["failing"][w_index])
                r["interaction_passing_minus_failing"][label+"_ci95"]=interval(draws).T
            r["readability_passing_minus_failing"]={}
            for method in METHODS:
                values=s[method]["any_excess"]
                draws=boot_means(values[mask],stratum_weights["passing"][0])-boot_means(values[~mask],stratum_weights["failing"][0])
                r["readability_passing_minus_failing"][method]={"mean":values[mask].mean(axis=0)-values[~mask].mean(axis=0),"ci95":interval(draws).T}
            result["criteria"][c][task]=r
    save_json(out/"correctness.json",result)
    save_csv(out/"correctness.csv",rows)
    fig,axes=plt.subplots(2,2,figsize=(11,7.5),layout="constrained")
    x=np.arange(1,5)
    for ti,task in enumerate(TASKS):
        r=result["criteria"]["historical"][task]
        for si,(label,style) in enumerate((("passing","-"),("failing","--"))):
            for mi,method in enumerate(METHODS):
                z=r["strata"][label]["methods"][method]["any_excess"]
                offset=(si*2+mi-1.5)*.065
                axes[ti,0].errorbar(x+offset,z["mean"],yerr=np.array([z["mean"]-z["ci95"][:,0],z["ci95"][:,1]-z["mean"]]),
                    label=f"{METHOD_LABELS[method]} · {label}",color=COLORS[method],ls=style,marker="o" if si==0 else "s",ms=4,capsize=2)
            z=r["strata"][label]["any_excess_difference"]
            axes[ti,1].errorbar(x+(si-.5)*.1,z["mean"],yerr=np.array([z["mean"]-z["ci95"][:,0],z["ci95"][:,1]-z["mean"]]),
                label=f"{label} · n={r['n_'+label]}",color="#357b60" if si==0 else "#8a597a",ls=style,marker="o",capsize=3)
        for j in range(2):
            axes[ti,j].axhline(0,color="#555",lw=.7);axes[ti,j].set_xticks(x);axes[ti,j].set_xlabel("Recurrent loop")
            axes[ti,j].set_title(f"{TASK_LABELS[task]} · {r['n_passing']} passing / {r['n_failing']} failing")
            axes[ti,j].legend(fontsize=8)
        axes[ti,0].set_ylabel("Any-layer excess hit@10");axes[ti,1].set_ylabel("Paired J-Lens − logit lens")
    fig.suptitle("Ouro · historical task criterion · paired item 95% intervals\nPassing is the original answer check; corrected numerical sensitivities are in the tables")
    plot_finish(fig,out,"correctness_stratified")
    # Full depth structure for each stratum; avoid only reporting any-layer.
    fig,axes=plt.subplots(2,4,figsize=(15,6.5),sharex=True,sharey=True,layout="constrained")
    for ti,task in enumerate(TASKS):
        r=result["criteria"]["historical"][task]
        for u in range(4):
            ax=axes[ti,u]
            for label,color in (("passing","#357b60"),("failing","#8a597a")):
                z=r["strata"][label]["excess_difference"]
                ax.plot(np.arange(1,49),z["mean"][u],color=color,label=f"{label} n={r['n_'+label]}")
                ax.fill_between(np.arange(1,49),z["ci95"][u,:,0],z["ci95"][u,:,1],color=color,alpha=.13)
            ax.axhline(0,color="#555",lw=.7);ax.set_title(f"{TASK_LABELS[task]} · loop {u+1}")
            ax.set_xticks([1,16,32,48]);ax.set_xlabel("Physical layer")
            if u==0:ax.set_ylabel("J-Lens − logit lens");ax.legend(fontsize=8)
    fig.suptitle("Correctness-stratified layer curves · excess hit@10\nHistorical criterion; pointwise paired item intervals, not a search for significant subgroups")
    plot_finish(fig,out,"correctness_layerwise_difference")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=HERE / "results")
    parser.add_argument("--part",choices=["layers","exits","correctness","all"],default="all")
    args=parser.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    items,names,arrays=load_run(MAIN)
    if args.part in ("layers","all"):
        layer_analysis(items,names,arrays,out)
    if args.part in ("exits","all"):
        exit_analysis(out)
    if args.part in ("correctness","all"):
        correctness_analysis(items,names,arrays,out)
    save_json(out/"inputs.json",{"generator":str(Path(__file__)),"generator_sha256":digest(__file__),
              "input_files":[{"path":str(p),"sha256":digest(p)} for root in (MAIN,LOCAL,POS2)
                             for p in (root/"arrays.npz",root/"items.json",root/"task_names.json",root/"provenance.json")],
              "numpy":np.__version__,"matplotlib":matplotlib.__version__,"bootstrap_draws":N_BOOT})


if __name__=="__main__":
    main()
