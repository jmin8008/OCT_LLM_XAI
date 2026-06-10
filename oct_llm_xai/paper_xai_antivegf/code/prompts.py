"""Prompt templates (Z0/Z1/F2) and response parsing for anti-VEGF VLM inference.

Templates follow EXPERIMENTAL_PROTOCOL.md §2.1:
  Z0 - zero-shot direct continue/stop
  Z1 - biomarker-guided (state IRF/SRF/PED first, then decide)
  F2 - few-shot (2-4 labelled exemplars prepended)

Parsing is regex-based and conservative: a CI answer that cannot be resolved to
continue/stop becomes "uncertain" and is tallied separately (never silently
coerced). KG triples can be injected as RAG context (see build_prompt kg_context).
"""
from __future__ import annotations

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------
Z0 = (
    "This is a pre-treatment macular OCT B-scan of a patient with {diagnosis}. "
    "Based on the retinal fluid and structure, will this patient need continued "
    "anti-VEGF injections? Answer 'continue' or 'stop' and explain."
)

Z1 = (
    "This is a pre-treatment macular OCT B-scan of a patient with {diagnosis}. "
    "First state whether IRF (intraretinal fluid), SRF (subretinal fluid), and "
    "PED (pigment epithelial detachment) are present. Then decide: will this "
    "patient need continued anti-VEGF injections? Answer 'continue' or 'stop' "
    "and explain."
)

F2_HEADER = (
    "You are an ophthalmology assistant deciding anti-VEGF continuation from OCT. "
    "Here are labelled examples:\n{exemplars}\n\nNow the new case:\n"
)

BIOMARKER_PROMPT = (
    "This is a macular OCT B-scan. For each of the following, answer present or "
    "absent: IRF (intraretinal fluid), SRF (subretinal fluid), PED (pigment "
    "epithelial detachment), HRF (hyperreflective foci)."
)

# JSON-forced variant: each biomarker MUST be exactly one of present/absent
# (never both) so the value is unambiguous to parse. Used for E2/E7.
BIOMARKER_PROMPT_JSON = (
    "This is a macular OCT B-scan. Decide for each biomarker whether it is present "
    "or absent. For each one choose EXACTLY ONE word — either \"present\" or "
    "\"absent\", never both, never uncertain. Respond with ONLY a JSON object and "
    "no other text:\n"
    '{"IRF": "<present|absent>", "SRF": "<present|absent>", '
    '"PED": "<present|absent>", "HRF": "<present|absent>"}'
)

# Z2: KG chain-of-thought — explicit step-by-step reasoning aligned with guideline rules
Z2_KG_COT = (
    "This is a pre-treatment macular OCT B-scan of a patient with {diagnosis}.\n\n"
    "Follow these steps:\n"
    "Step 1 — Identify biomarkers. Report ONLY as JSON:\n"
    '  {{"IRF": "<present|absent>", "SRF": "<present|absent>", '
    '"PED": "<present|absent>", "HRF": "<present|absent>"}}\n\n'
    "Step 2 — Apply clinical guidelines:\n"
    "  - If IRF present → clinical evidence for continuing anti-VEGF\n"
    "  - If SRF present → clinical evidence for continuing anti-VEGF\n"
    "  - If both IRF and SRF absent (dry macula) → clinical evidence for stopping\n"
    "  - If PED only, no active fluid → case-dependent\n\n"
    "Step 3 — State your final decision: 'continue' or 'stop'. "
    "Answer 'continue' or 'stop' and explain which guideline rule you applied."
)

# AMD staging task (used for catastrophic forgetting test — RetinaVLM's trained task)
AMD_STAGING_QUERY = (
    "Describe the OCT image in detail and list any biomarkers or abnormalities, "
    "including the most likely AMD stage of the patient. "
    "State if the patient's most advanced AMD stage is "
    "'healthy', 'early', 'intermediate', 'late dry', 'late wet (inactive)' or 'late wet (active)'."
)
AMD_STAGING_STEP2 = "Based on the image and those findings, the patient's most advanced AMD stage is"

# General VQA — forgetting baseline (any model should handle this)
GENERAL_VQA = "What type of medical image is this? Briefly describe what you observe."

VA_PROMPT = "Estimate the post-treatment visual acuity (decimal, e.g. 0.5) for this eye."
CST_PROMPT = (
    "Estimate the post-treatment central subfield thickness in microns (a number) "
    "for this eye."
)

KG_CONTEXT_HEADER = (
    "\n\nClinical guideline knowledge (anti-VEGF continuation rules):\n{rules}\n"
    "Use these rules where the imaging supports them.\n"
)


def build_prompt(
    diagnosis: str,
    variant: str = "Z0",
    kg_context: Optional[str] = None,
    exemplars: Optional[str] = None,
    kg=None,          # GuidelineKG instance, used for Z2_KG_COT context injection
) -> str:
    """Compose the CI prompt for a variant, optionally with RAG KG context.

    kg_context: pre-rendered guideline rule lines (from kg.render_rag_context).
    exemplars : pre-rendered few-shot block (required for F2).
    """
    if variant == "Z0":
        body = Z0.format(diagnosis=diagnosis)
    elif variant == "Z1":
        body = Z1.format(diagnosis=diagnosis)
    elif variant == "Z2_KG_COT":
        body = Z2_KG_COT.format(diagnosis=diagnosis)
    elif variant == "F2":
        head = F2_HEADER.format(exemplars=exemplars or "(no exemplars provided)")
        body = head + Z0.format(diagnosis=diagnosis)
    else:
        raise ValueError(f"unknown variant {variant!r}")
    if kg_context:
        body += KG_CONTEXT_HEADER.format(rules=kg_context)
    return body


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
_STOP_RE = re.compile(r"\b(stop|discontinue|cease|no\s+(?:further|more)\s+injection)\b", re.I)
_CONT_RE = re.compile(r"\b(continue|continued|continuation|keep\s+inject|further\s+injection)\b", re.I)


def parse_ci(text: str):
    """Return 1 (continue), 0 (stop), or 'uncertain'.

    Resolves conflicts by taking the answer nearest an explicit 'answer:' cue,
    else the first decisive keyword; ambiguous/none -> 'uncertain'.
    """
    if not text:
        return "uncertain"
    # Authoritative explicit tag wins; take the LAST 'Decision:'/'Answer:' so the
    # conclusive call is read even when the reasoning mentions the other option first
    # (e.g. divergent "...would favor continuation ... was to stop. Decision: stop").
    tags = re.findall(r"(?:decision|answer)\s*[:\-]?\s*(continue|stop)", text, re.I)
    if tags:
        return 1 if tags[-1].lower() == "continue" else 0
    has_cont = bool(_CONT_RE.search(text))
    has_stop = bool(_STOP_RE.search(text))
    if has_cont and not has_stop:
        return 1
    if has_stop and not has_cont:
        return 0
    if has_cont and has_stop:
        # take whichever appears first
        return 1 if _CONT_RE.search(text).start() < _STOP_RE.search(text).start() else 0
    return "uncertain"


def parse_response(text: str):
    """Step-4 composite treatment-response node (v0.3): returns
    'good_responder' | 'poor_responder' | 'no_active_disease' | None.

    Reads the Step-4 segment only; 'no active disease' wins (dry/quiescent), else the
    LAST explicit good/poor cue is the conclusion (narratives may mention 'poor' while
    concluding 'good' or vice-versa)."""
    if not text:
        return None
    t = text.lower()
    if "step 4" in t:
        t = t.split("step 4", 1)[1]
    if "no active disease" in t or "no active exudation" in t:
        return "no_active_disease"
    good_i = max(t.rfind("good respon"), t.rfind("good treatment respon"))
    poor_i = max(t.rfind("poor respon"), t.rfind("non-responder"),
                 t.rfind("response to anti-vegf was poor"), t.rfind("limited benefit"),
                 t.rfind("guarded prognosis"))
    if good_i < 0 and poor_i < 0:
        return None
    return "good_responder" if good_i > poor_i else "poor_responder"


# Absent cues are checked FIRST (negations like "not seen" must win over the bare
# "seen"/"noted" present cues they contain).
_ABSENT_RE = r"(absent|negative|not\s+(?:present|seen|noted|detected)|no\b|none|without|\-)"
_PRESENT_RE = r"(present|positive|seen|noted|detected|identified|yes|\+)"


def parse_biomarkers(text: str) -> dict:
    """Return {IRF/SRF/PED/HRF: 1/0/nan} from free text."""
    import math

    names = {
        "IRF": r"(IRF|intraretinal\s+fluid)",
        "SRF": r"(SRF|subretinal\s+fluid)",
        "PED": r"(PED|pigment\s+epithelial\s+detachment)",
        "HRF": r"(HRF|hyperreflective\s+foci)",
    }
    neg_before = r"(no|without|absent|negative\s+for)\s+"
    out = {}
    for key, pat in names.items():
        # negation immediately BEFORE the name ("No subretinal fluid", "without SRF")
        if re.search(neg_before + pat, text, re.I):
            out[key] = 0
            continue
        # cue AFTER the name within a short window (stops at sentence punctuation)
        m = re.search(pat + r"[^.\n;:]{0,40}?" + f"(?:{_ABSENT_RE}|{_PRESENT_RE})", text, re.I)
        if not m:
            out[key] = math.nan
            continue
        seg = m.group(0).lower()
        if re.search(_ABSENT_RE, seg, re.I):       # absent priority (handles "not seen")
            out[key] = 0
        elif re.search(_PRESENT_RE, seg, re.I):
            out[key] = 1
        else:
            out[key] = math.nan
    return out


def parse_biomarkers_json(text: str) -> dict:
    """Parse a JSON biomarker object (present/absent -> 1/0). Falls back to the
    free-text parser when no valid JSON is found (e.g. RetinaVLM prose)."""
    import json as _json
    import math

    out = {k: math.nan for k in ("IRF", "SRF", "PED", "HRF")}
    m = re.search(r"\{[^{}]*\}", text or "", re.S)
    if m:
        try:
            d = _json.loads(m.group(0))
            hit = False
            for k in out:
                v = str(d.get(k, "")).strip().lower()
                if v.startswith("present") or v in ("1", "true", "yes"):
                    out[k] = 1; hit = True
                elif v.startswith("absent") or v in ("0", "false", "no"):
                    out[k] = 0; hit = True
            if hit:
                return out
        except (ValueError, TypeError):
            pass
    return parse_biomarkers(text)   # prose fallback


_NUM_RE = re.compile(r"(\d+(?:\.\d+)?)")


def parse_number(text: str, lo: float = -1e9, hi: float = 1e9):
    """First number in range, else nan."""
    import math

    for tok in _NUM_RE.findall(text or ""):
        v = float(tok)
        if lo <= v <= hi:
            return v
    return math.nan


def parse_va(text: str):
    return parse_number(text, lo=0.0, hi=2.0)


def parse_cst(text: str):
    return parse_number(text, lo=100.0, hi=1200.0)


if __name__ == "__main__":
    print(build_prompt("DME", "Z1", kg_context="- SRF present -> continue (w=0.85)"))
    print("CI:", parse_ci("The scan shows SRF. Answer: continue with injections."))
    print("CI:", parse_ci("Macula is dry, recommend to stop."))
    print("CI:", parse_ci("It is unclear."))
    print("BM:", parse_biomarkers("IRF present, SRF absent, PED not seen, HRF positive"))
    print("VA:", parse_va("Estimated post-treatment VA is about 0.5 decimal"))
    print("CST:", parse_cst("post-treatment CST approximately 320 microns"))
