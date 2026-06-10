"""End-to-end verification of the v0.3 CoT/KG/metadata/SFT pipeline.

Audits 4 things the user asked to confirm:
  A. CoT structure   — every target has the reordered Step1-4 (decision before response)
  B. KG consistency  — nodes/edges/layers valid; generator narratives come FROM the KG
  C. metadata lineage— each GT field traced to its source (metadata_v2 + pic CSV); the
                       INPUT prompt leaks NO post/outcome info; meta-ablation injects
                       only PRE-treatment fields
  D. SFT integrity   — schema, counts, GT<->source consistency, CF pairing, images

Run: PYTHONNOUSERSITE=1 conda run -n aptos2021 python3 -u verify_v3.py
"""
from __future__ import annotations
import json, os, re

ROOT = "/home/ubuntu/bionexus/jgy/OCT_LLM_XAI/oct_llm_xai/paper_xai_antivegf"
SFT = f"{ROOT}/sft_data/sft_kg_cot.json"
META = f"{ROOT}/fluid_masks_v2/metadata_v2.json"
KG = f"{ROOT}/code/antivegf_guideline_kg_v2.json"
PIC = ("/home/ubuntu/bionexus/jgy/OCT_LLM_XAI/data_response/anti-vegf-dataset/"
       "APTOS-2021/Final Datasets/train_anno_pic.csv")
ok = fail = 0


def check(name, cond, detail=""):
    global ok, fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    ok += cond; fail += (not cond)


def main():
    rows = json.load(open(SFT))
    meta = {m["eye_id"]: m for m in json.load(open(META))}
    kg = json.load(open(KG))
    fac = [r for r in rows if r["type"] == "factual"]
    cf = [r for r in rows if r["type"] == "counterfactual"]

    print("\n=== A. CoT 구조 ===")
    hdr = ["Step 1 (Visual findings):", "Step 2 (Pathophysiology):",
           "Step 3 (Clinical decision):", "Step 4 (Predicted treatment response):"]
    bad_order = [r["eye_id"] for r in fac
                 if not all(h in r["target"] for h in hdr)
                 or not (r["target"].index(hdr[2]) < r["target"].index(hdr[3]))]
    check("모든 factual에 Step1-4 + decision(3) before response(4)", not bad_order, f"위반 {bad_order[:3]}")
    no_dx = all("patient with" not in r["prompt"] and "DME" not in r["prompt"]
                and "PCV" not in r["prompt"] and "CNVM" not in r["prompt"] for r in rows)
    check("입력 프롬프트에 진단명 없음", no_dx)
    dec_in_s3 = all(re.search(r"Decision:\s*(continue|stop)", r["target"].split("Step 4")[0]) for r in fac)
    check("Decision 태그가 Step3(=응답 앞)에 위치", dec_in_s3)

    print("\n=== B. KG 일관성 + 실제 사용 ===")
    node_ids = {n["id"] for n in kg["nodes"]}
    node_layers = {n["layer"] for n in kg["nodes"]}
    edge_ok = all(e["head"] in node_ids and e["tail"] in node_ids for e in kg["edges"])
    check("모든 엣지 head/tail이 유효 노드", edge_ok)
    check("layers == 선언된 4계층", node_layers == set(kg["layers"]) == {"visual","pathophysiology","decision","response"})
    # decision -> response 방향(요청 핵심)
    dr = [e for e in kg["edges"] if e["head"] in ("continue","stop")]
    check("decision->response 엣지 존재(인과 방향)", len(dr) >= 2 and all(e["tail"] in node_ids for e in dr),
          f"{[(e['head'],e['tail']) for e in dr]}")
    # 생성기가 KG narrative를 실제로 사용하는지: target 안에 KG 노드 narrative가 등장
    narr = {n["id"]: n.get("narrative", n["label"]) for n in kg["nodes"]}
    used = sum(narr["active_exudation"] in r["target"] for r in fac)
    check("target이 KG 노드 narrative를 인용(active_exudation)", used > 0, f"{used} eyes")
    check("KG에 response_definition(composite rule) 존재", "response_definition" in kg and "composite_responder_rule" in kg["response_definition"])

    print("\n=== C. metadata lineage (어디서 끌어오나) ===")
    # pre-fluid + post-fluid 교차검증
    try:
        import pandas as pd
        df = pd.read_csv(PIC); df.columns = [c.strip() for c in df.columns]
        df["inj"] = df["injection"].astype(str).str.lower()
        post = df[df.inj.str.startswith("post")].groupby("patient ID")[["IRF","SRF"]].max()
        post_fl = {e: bool(r["IRF"] or r["SRF"]) for e, r in post.iterrows()}
    except Exception as e:
        post_fl = {}; print("   (pandas/pic CSV 미사용)", str(e)[:60])
    # 표본 3개 eye에 대해 GT 필드 출처 추적
    sample = [r for r in fac if r["eye_id"] in ("2L", "20R", "5L")] or fac[:3]
    lineage_ok = True
    for r in sample:
        e = r["eye_id"]; m = meta[e]; g = r["nodes_gt"]
        dec_src = "continue" if m["continue_injection"] == 1 else "stop"
        fr_src = ("na" if e not in post_fl else ("persistent" if post_fl[e] else "resolved"))
        good = (g["decision"] == dec_src and g["delta_cst"] == m["delta_cst"]
                and g["biomarkers"]["IRF"] == int(m["biomarkers"].get("IRF", 0))
                and g["fluid_resolution"] == fr_src)
        lineage_ok &= good
        print(f"   eye {e}: decision={g['decision']}<-label{dec_src} | ΔCST={g['delta_cst']}<-meta | "
              f"fluid_res={g['fluid_resolution']}<-picCSV({fr_src}) | resp={g['response']}")
    check("GT 필드가 출처(metadata_v2 label/ΔCST/biomarker + pic CSV post-fluid)와 일치", lineage_ok)
    # 입력 프롬프트 outcome 누설 검사 — 올바른 기준: base 프롬프트가 eye-독립(generic)이면
    # per-eye outcome(ΔCST 값, post, 실제 responder 라벨)을 흘릴 방법이 없음. (Step4 지시문의
    # 'responder/resolution'는 과제 설명이지 정답이 아님.)
    base_prompts = set(r["prompt"] for r in fac)
    check("base 프롬프트가 eye-독립(generic, 1종) → per-eye outcome 누설 불가", len(base_prompts) == 1,
          f"{len(base_prompts)} unique")
    # 어떤 eye의 실제 outcome 값도 그 eye 프롬프트에 없음 (word-boundary로 'Step 3/4' 숫자 오탐 방지;
    # post_cst는 3자리라 안전)
    no_val_leak = all(not re.search(rf"\b{int(meta[r['eye_id']]['post_cst'])}\b", r['prompt']) for r in fac)
    check("어떤 eye 프롬프트에도 그 eye의 post_cst 값 없음", no_val_leak)
    # 메타 ablation이 주입하는 필드 = pre만 (주입된 'Patient context' 절만 따로 검사)
    import sys; sys.path.insert(0, f"{ROOT}/code")
    import harness, copy
    rr = copy.deepcopy([fac[0]]); harness.apply_meta(rr)
    inj = rr[0]["prompt"].split("Patient context:")[1].split("\nReason step by step:")[0].lower()
    inj_pre = all(s in inj for s in ("age", "baseline visual acuity", "baseline central"))
    inj_nopost = not any(s in inj for s in ("post", "δcst", "delta", "responder", "reduction", "resolution"))
    check("meta-주입 절=pre만(age/gender/drug/preVA/preCST), post/outcome 없음", inj_pre and inj_nopost,
          f"injected='{inj.strip()[:80]}...'")

    print("\n=== D. SFT 무결성 ===")
    schema = ["id","eye_id","split","type","image","prompt","target","nodes_gt"]
    check("스키마 완비", all(all(k in r for k in schema) for r in rows))
    check("행수 factual=218, cf=211, 총=429", len(fac) == 218 and len(cf) == 211 and len(rows) == 429)
    # composite responder 규칙 재검증
    def comp(d, v): return "good_responder" if ((d is not None and d <= -25) or (v is not None and v >= 0.1)) else "poor_responder"
    resp_ok = all(r["nodes_gt"]["response"] == comp(r["nodes_gt"]["delta_cst"], r["nodes_gt"]["delta_va"])
                  for r in fac if r["nodes_gt"]["response"] != "no_active_disease")
    check("response GT == composite rule(ΔCST≤-25 OR ΔVA≥0.1)", resp_ok)
    # CF pairing: 모든 cf eye는 factual continue/stop 짝 + target은 stop/no_active
    fac_eyes = {r["eye_id"] for r in fac}
    cf_pair = all(r["eye_id"] in fac_eyes for r in cf)
    cf_stop = all(r["nodes_gt"]["decision"] == "stop" and r["nodes_gt"]["response"] == "no_active_disease" for r in cf)
    check("CF 짝 존재 + CF는 stop/no_active_disease", cf_pair and cf_stop)
    # 이미지 존재
    miss = [r["eye_id"] for r in rows if not os.path.exists(f"{ROOT}/{r['image']}")]
    check("모든 참조 이미지 존재", not miss, f"누락 {miss[:3]}")
    # split 누수 없음 (train/test eye 분리)
    tr = {r["eye_id"] for r in rows if r["split"] == "train"}
    te = {r["eye_id"] for r in rows if r["split"] == "test"}
    check("train/test eye 누수 없음", not (tr & te), f"교집합 {list(tr & te)[:3]}")

    print(f"\n=== 결과: {ok} PASS / {fail} FAIL ===")


if __name__ == "__main__":
    main()
