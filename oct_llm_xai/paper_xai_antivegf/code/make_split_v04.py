"""v0.4 eye-level 3-way split (train/val/test) with the Core-only Quota gate.

Why this exists (vs data.py:stratified_split):
  - stratum key is decision x responder x has_fluid (not diagnosis x continue),
  - 3-way (train/val/test) instead of 2-way,
  - dry eyes (has_fluid=0, only 7) are pooled into one group so the ratio split
    cannot starve the test set of them (per-stratum ratio would round 1->0),
  - a Core-only minimum-inclusion gate on TEST: dry>=2 (sole hard constraint),
    stop>=1, poor>=1 (symbolic floors). Retries seeds until the gate passes.

Output: sft_data/split_v04.json  {eye_id: "train"|"val"|"test", _meta:{...}}.
Does NOT touch sft_kg_cot.json (re-gen + retrain is the later GPU step).

Run (no GPU, stdlib+json only):
  python3 -u make_split_v04.py            # default test 0.32 / val 0.10
  python3 -u make_split_v04.py 0.30 0.10  # custom test_frac val_frac
"""
from __future__ import annotations
import json, os, random, sys
from collections import Counter

ROOT = "/home/ubuntu/bionexus/jgy/OCT_LLM_XAI/oct_llm_xai/paper_xai_antivegf"
META = f"{ROOT}/fluid_masks_v2/metadata_v2.json"
OUT = f"{ROOT}/sft_data/split_v04.json"

# Core-only Quota (TEST set) — see EXPERIMENTAL_PROTOCOL.md §4.2
QUOTA = {"dry": 2, "stop": 1, "poor": 1}


def has_fluid(bm):
    return bool(bm.get("IRF") or bm.get("SRF"))


def responder(dcst, dva, bm):
    if not (has_fluid(bm) or bm.get("PED")):
        return "no_active"
    anat = (dcst is not None and dcst <= -25)
    func = (dva is not None and dva >= 0.1)
    return "good" if (anat or func) else "poor"


def decision(r):
    return "continue" if r["continue_injection"] == 1 else "stop"


def assert_min_inclusion(test_rows):
    """Core-only gate. Returns (ok, counts)."""
    c = {
        "dry": sum(1 for r in test_rows if not has_fluid(r["biomarkers"])),
        "stop": sum(1 for r in test_rows if decision(r) == "stop"),
        "poor": sum(1 for r in test_rows if responder(r["delta_cst"], r["delta_va"], r["biomarkers"]) == "poor"),
    }
    ok = all(c[k] >= QUOTA[k] for k in QUOTA)
    return ok, c


def assign_group(members, test_frac, val_frac, rng):
    """3-way assign one stratum: first n_test->test, next n_val->val, rest->train."""
    m = list(members)
    rng.shuffle(m)
    n = len(m)
    n_test = round(n * test_frac)
    n_val = round(n * val_frac)
    return {"test": m[:n_test], "val": m[n_test:n_test + n_val], "train": m[n_test + n_val:]}


def make_split(meta, test_frac, val_frac, seed):
    # dry pool (7 eyes) handled as ONE group so test can't round to 0;
    # wet eyes stratified by decision x responder.
    dry = [r for r in meta if not has_fluid(r["biomarkers"])]
    wet = [r for r in meta if has_fluid(r["biomarkers"])]
    groups = {("__dry__",): dry}
    for r in wet:
        key = (decision(r), responder(r["delta_cst"], r["delta_va"], r["biomarkers"]))
        groups.setdefault(key, []).append(r)

    rng = random.Random(seed)
    split = {}
    for key in sorted(groups, key=str):
        a = assign_group(groups[key], test_frac, val_frac, rng)
        for s in ("train", "val", "test"):
            for r in a[s]:
                split[r["eye_id"]] = s
    return split


def report(meta, split):
    by = {"train": [], "val": [], "test": []}
    for r in meta:
        by[split[r["eye_id"]]].append(r)
    print(f"{'split':6} {'n':>4} {'cont%':>6} {'good%':>6} {'progMaj':>7} {'dry':>4} {'stop':>5} {'poor':>5} | IRF/SRF/PED present%")
    for s in ("train", "val", "test"):
        rows = by[s]
        n = len(rows)
        cont = sum(decision(r) == "continue" for r in rows) / n
        rp = Counter(responder(r["delta_cst"], r["delta_va"], r["biomarkers"]) for r in rows)
        gp = rp["good"] + rp["poor"]
        prog = Counter()
        for r in rows:
            d = r["delta_cst"]
            b = ("marked" if d <= -100 else "partial" if d <= -25 else "minimal" if d <= 25 else "worsening") if d is not None else None
            prog[b] += 1
        maj = max(prog.values()) / n
        dry = sum(1 for r in rows if not has_fluid(r["biomarkers"]))
        stop = sum(1 for r in rows if decision(r) == "stop")
        poor = rp["poor"]
        bmp = {k: sum(int(r["biomarkers"].get(k, 0) == 1) for r in rows) / n for k in ("IRF", "SRF", "PED")}
        print(f"{s:6} {n:>4} {cont:>6.3f} {rp['good']/max(gp,1):>6.3f} {maj:>7.3f} {dry:>4} {stop:>5} {poor:>5} | "
              f"{bmp['IRF']:.2f}/{bmp['SRF']:.2f}/{bmp['PED']:.2f}")
    return by


def main():
    test_frac = float(sys.argv[1]) if len(sys.argv) > 1 else 0.32
    val_frac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.10
    meta = json.load(open(META))
    print(f"[make_split_v04] {len(meta)} eyes | test_frac={test_frac} val_frac={val_frac} | QUOTA(test)={QUOTA}")

    split = None
    for seed in range(200):
        cand = make_split(meta, test_frac, val_frac, seed)
        test_rows = [r for r in meta if cand[r["eye_id"]] == "test"]
        ok, counts = assert_min_inclusion(test_rows)
        if ok:
            print(f"  seed {seed}: PASS gate {counts}")
            split = cand
            chosen_seed = seed
            break
        else:
            if seed < 5:
                print(f"  seed {seed}: FAIL gate {counts} (need {QUOTA}) -> retry")
    if split is None:
        print("ERROR: no seed satisfied the Core-only Quota in 200 tries -> relax frac/quota")
        sys.exit(1)

    by = report(meta, split)
    out = {"_meta": {"version": "v0.4", "seed": chosen_seed, "test_frac": test_frac,
                     "val_frac": val_frac, "quota": QUOTA,
                     "counts": {s: len(by[s]) for s in by}}}
    out.update(split)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}  (train/val/test = {len(by['train'])}/{len(by['val'])}/{len(by['test'])})")
    # per-slice negative coverage estimate for the chosen test eyes (biomarker regime)
    print("note: per-slice biomarker eval runs on these test-eye slices (no train leakage).")


if __name__ == "__main__":
    main()
