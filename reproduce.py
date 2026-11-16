"""Recompute the paper's tables from pooled detector and probe scores."""
import argparse
from bisect import bisect_right
from collections import defaultdict
import csv
import gzip
from itertools import product, repeat
from math import floor
from pathlib import Path
from statistics import mean

DATA = Path(__file__).resolve().parent / "data"
MODELS = ("Qwen", "Mistral")
TASKS = ("SST-2", "Rotten Tomatoes", "TREC", "Jigsaw", "AG News")
ATTACKS = ("BadNets", "AddSent", "EP", "SOS")
PROMPTS = ("Reasoning", "ZS-CoT", "Auto-CoT")
METHODS = ("GD", "ONION", "BBCaL-FPS", "CoS", "CoS-original", *PROMPTS)


def quantile(values, p):
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    i = floor(position)
    lower, upper = ordered[i], ordered[min(i + 1, len(ordered) - 1)]
    fraction = position - i
    if fraction >= 0.5:
        return upper - (upper - lower) * (1 - fraction)
    return lower + (upper - lower) * fraction


def auc(bins):
    clean_before = twice_wins = total_attack = 0
    for score, clean, attack, *_ in sorted(bins):
        twice_wins += attack * (2 * clean_before + clean)
        clean_before += clean
        total_attack += attack
    return twice_wins / (2 * clean_before * total_attack)


def metrics(bins, budget, reject_ties=False):
    clean = [score for score, n, *_ in bins for _ in range(n)]
    attacks = sum(b[2] for b in bins)
    if (len(clean), attacks) != (200, 300):
        raise ValueError("Expected 200 clean and 300 triggered evaluation inputs")
    cutoff = quantile(clean, 1 - budget)
    accepted = [b for b in bins if b[0] < cutoff or (not reject_ties and b[0] == cutoff)]
    rasr = sum(b[4] for b in accepted) / attacks
    fpr = (len(clean) - sum(b[1] for b in accepted)) / len(clean)
    return auc(bins), rasr, fpr


def write(folder, name, fields, rows):
    with (folder / name).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fields.split())
        writer.writerows(rows)


def averaged(rows):
    return tuple(map(mean, zip(*rows)))


def main_benchmark(out):
    scores = defaultdict(list)
    with gzip.open(DATA / "main_scores.csv.gz", "rt", newline="") as f:
        for r in csv.DictReader(f):
            key = r["model"], r["dataset"], r["attack"], r["method"], int(r["seed"])
            scores[key].append((float(r["score"]), int(r["clean"]), int(r["triggered"]),
                                int(r["clean_correct"]), int(r["target_success"])))
    if len(scores) != 960:
        raise ValueError("Expected 120 victim runs and eight detectors per run")
    full, main, validity = [], [], []
    wins_auc = wins_rasr = 0
    for model, task, attack in product(MODELS, TASKS, ATTACKS):
        case = model, task, attack
        results = {}
        for method in METHODS:
            seeds = [scores[(*case, method, seed)] for seed in range(3)]
            results[method] = averaged([metrics(b, .05) for b in seeds])
            full.append((*case, method, *results[method]))
            if method == "GD":
                accuracy = mean(sum(b[3] for b in run) / 200 for run in seeds)
                pre_asr = mean(sum(b[4] for b in run) / 300 for run in seeds)
                validity.append((*case, accuracy, pre_asr))
        def reasoning_order(method):
            auroc, residual, fpr = results[method]
            if model == "Qwen":
                return residual, -auroc, fpr
            return residual, fpr, -auroc

        prompt = min(PROMPTS, key=reasoning_order)
        for method in ("GD", "ONION", "BBCaL-FPS", "CoS", prompt):
            shown = "Reasoning-best" if method == prompt else method
            main.append((*case, shown, method, *results[method]))
        others = [v for m,v in results.items() if m != "GD"]
        wins_auc += results["GD"][0] > max(v[0] for v in others)
        wins_rasr += results["GD"][1] < min(v[1] for v in others)
    write(out, "table1.csv", "model dataset attack column method auroc rasr clean_fpr", main)
    write(out, "table6.csv", "model dataset attack clean_accuracy pre_asr", validity)
    write(out, "tables13-17.csv", "model dataset attack method auroc rasr clean_fpr", full)
    return wins_auc, wins_rasr


def ranks(reference, value, add_one=False):
    return (int(add_one) + bisect_right(reference, value)) / (int(add_one) + len(reference))


def by_label(rows, field, label, minimum):
    selected = [r for r in rows if r[field] == label]
    return selected if len(selected) >= minimum else rows


def probe_bins(rows, values):
    counts = defaultdict(lambda: [0, 0, 0, 0])
    for r, score in zip(rows, values, strict=True):
        group = int(r["triggered"])
        counts[score][group] += 1
        counts[score][2 + group] += int(r["correct"])
    return [(s, *v) for s,v in counts.items()]


def stress(cal, ev):
    gd_ref = sorted(r["gd"] for r in cal)
    joint = []
    for label in {r["ug_label"] for r in ev}:
        pool = by_label(cal, "ug_label", label, 10)
        refs = {k: sorted(r[k] for r in pool) for k in ("support", "document")}
        for r in ev:
            if r["ug_label"] == label:
                component = max(ranks(refs[k], r[k]) for k in refs)
                joint.append((r, max(ranks(gd_ref, r["gd"]), component)))
    grounding = probe_bins(ev, [r["gd"] for r in ev])
    combined = probe_bins([r for r,_ in joint], [s for _,s in joint])
    result = {}
    for name, bins in (("Grounding", grounding), ("Joint", combined)):
        auroc, r10, f10 = metrics(bins, .10, reject_ties=True)
        _, r20, f20 = metrics(bins, .20, reject_ties=True)
        result[name] = (auroc, r10, r20, f10, f20)
    return result


def verifier(cal, ev, budget):
    gd_cut = {label: quantile([r["gd"] for r in by_label(cal, "gd_label", label, 4)], .95)
              for label in {r["gd_label"] for r in cal + ev}}
    pool = [r for r in cal if r["gd"] <= gd_cut[r["gd_label"]]]
    passed = [r for r in ev if r["gd"] <= gd_cut[r["gd_label"]]]
    beta = (budget - .05) / .95
    scored, kept = [], []
    for label in sorted({r["ug_label"] for r in passed}):
        ref = by_label(pool, "ug_label", label, 4)
        support = sorted(r["support"] for r in ref)
        document = sorted(r["document"] for r in ref)

        def fused(r):
            return max(ranks(support, r["support"], True), ranks(document, r["document"], True))

        threshold = quantile([fused(r) for r in ref], 1 - beta)
        for r in passed:
            if r["ug_label"] == label:
                score = fused(r)
                scored.append((r, score))
                if score <= threshold:
                    kept.append(r)
    bins = probe_bins([r for r,_ in scored], [s for _,s in scored])
    residual = sum(r["triggered"] and r["correct"] for r in kept) / 300
    fpr = (200 - sum(not r["triggered"] for r in kept)) / 200
    return auc(bins), residual, fpr


def probe_tables(out):
    groups = defaultdict(list)
    with gzip.open(DATA / "probe_scores.csv.gz", "rt", newline="") as f:
        for r in csv.DictReader(f):
            key = r["experiment"], r["attack"], int(r["seed"]), r["split"]
            for k in ("gd", "support", "document"):
                r[k] = float(r[k])
            for k in ("triggered", "correct"):
                r[k] = int(r[k])
            groups[key].extend(repeat(r, int(r["count"])))
    if len(groups) != 32:
        raise ValueError("Expected 12 stress runs and four paired verifier conditions")
    for (*_, split), records in groups.items():
        n_attack = sum(r["triggered"] for r in records)
        if (len(records), n_attack) != ((256, 0) if split == "calibration" else (500, 300)):
            raise ValueError("Unexpected calibration or evaluation counts")
    t2, t3 = [], []
    for attack in ATTACKS:
        seeds = [stress(groups["stress", attack, s, "calibration"],
                        groups["stress", attack, s, "evaluation"]) for s in range(3)]
        for policy in ("Grounding", "Joint"):
            t2.append((attack, policy, *averaged([r[policy] for r in seeds])))
    for attack, arm, budget in product(("BadNets", "SOS"), ("self", "independent"), (.10, .20)):
        cal = groups[arm, attack, 0, "calibration"]
        ev = groups[arm, attack, 0, "evaluation"]
        t3.append((attack, arm, budget, *verifier(cal, ev, budget)))
    write(out, "table2.csv", "attack score auroc rasr_10 rasr_20 fpr_10 fpr_20", t2)
    write(out, "table3.csv", "attack verifier budget ug_auroc rasr clean_fpr", t3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    wins = main_benchmark(args.output)
    probe_tables(args.output)
    print(f"GD has highest mean AUROC in {wins[0]}/40 and lowest mean rASR in {wins[1]}/40.")
