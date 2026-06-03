"""Neuro-symbolic AntiVEGF-Guideline-KG (Wang 2025, Sensors 25:6879 — adapted).

Lightweight, symbolic-first implementation:
  - directed labelled graph G=(V,E) with confidence-weighted causal edges,
  - production rules (IF biomarker-set THEN continue/stop) with
        rule_confidence = Π(premise_confidences) × rule_weight,
  - RAG context rendering for prompt injection (with/without-KG ablation),
  - decision-biomarker lookup for Attn-KG consistency (kg_align.py).

The neural entity encoder (PubMedBERT + InfoNCE, portable from the KAD notebook
experiments_product/5_kad_oct_diagnosis_executed.ipynb) is left as a STUB for a
future full neuro-symbolic extension; the paper uses the symbolic engine.
"""
from __future__ import annotations

import json
import os
from typing import Optional

_DEFAULT_KG = os.path.join(os.path.dirname(__file__), "antivegf_guideline_kg.json")

BIOMARKERS = ["IRF", "SRF", "PED", "HRF"]


class GuidelineKG:
    def __init__(self, spec: dict):
        self.spec = spec
        self.nodes = {n["id"]: n for n in spec["nodes"]}
        self.edges = spec["edges"]
        self.rules = spec["rules"]
        self.decision_biomarkers = spec.get("decision_biomarkers", {})

    @classmethod
    def load_default(cls) -> "GuidelineKG":
        return cls.load(_DEFAULT_KG)

    @classmethod
    def load(cls, path: str) -> "GuidelineKG":
        with open(path) as f:
            return cls(json.load(f))

    # -- symbolic forward-chaining ----------------------------------------
    def _rule_matches(self, rule: dict, biomarkers: dict) -> bool:
        for k, v in rule["if"].items():
            if biomarkers.get(k) != v:
                return False
        return True

    def forward_chain(self, biomarkers: dict) -> dict:
        """Apply production rules to a biomarker dict {IRF/SRF/PED/HRF: 0/1}.

        Returns {decision, confidence, fired_rules}. Confidence-weighted fusion:
        for each outcome we take the max rule_confidence among fired rules; the
        decision is the outcome with the highest aggregated confidence.

        Premise confidences come from the matching biomarker->outcome edge weight
        (defaulting to 1.0 when no edge), so
            rule_confidence = Π(premise_conf) × rule_weight.
        """
        edge_w = {(e["head"], e["tail"]): e["weight"] for e in self.edges}
        outcome_conf: dict[str, float] = {}
        fired = []
        for rule in self.rules:
            if not self._rule_matches(rule, biomarkers):
                continue
            premise_conf = 1.0
            for k, v in rule["if"].items():
                if v == 1:  # only "present" premises carry an edge weight
                    premise_conf *= edge_w.get((k, rule["then"]), 1.0)
            conf = premise_conf * rule["weight"]
            outcome_conf[rule["then"]] = max(outcome_conf.get(rule["then"], 0.0), conf)
            fired.append({"id": rule["id"], "then": rule["then"], "confidence": round(conf, 4)})
        if not outcome_conf:
            return {"decision": "uncertain", "confidence": 0.0, "fired_rules": []}
        decision = max(outcome_conf, key=outcome_conf.get)
        return {
            "decision": decision,
            "confidence": round(outcome_conf[decision], 4),
            "all_outcomes": {k: round(v, 4) for k, v in outcome_conf.items()},
            "fired_rules": fired,
        }

    # -- RAG context for prompt injection ---------------------------------
    def render_rag_context(self, biomarkers: Optional[dict] = None) -> str:
        """Render guideline rules as prompt context. If `biomarkers` given, only
        the relevant (matching/biomarker-mentioning) rules are returned (focused
        RAG); otherwise the full small rule set is returned."""
        lines = []
        for rule in self.rules:
            if biomarkers is not None and not any(
                k in rule["if"] for k in biomarkers
            ):
                continue
            lines.append(f"- {rule['rationale']} (confidence {rule['weight']})")
        return "\n".join(lines)

    def relevant_triples(self, entities) -> list[dict]:
        ents = set(entities)
        return [e for e in self.edges if e["head"] in ents or e["tail"] in ents]

    # -- helpers for kg_align.py ------------------------------------------
    def decision_drivers(self, decision: str) -> list[str]:
        """Biomarkers whose presence drives a given decision (for Attn-KG)."""
        return self.decision_biomarkers.get(decision, [])


# ---------------------------------------------------------------------------
# Neural entity encoder — STUB (future full neuro-symbolic extension).
# Port from experiments_product/5_kad_oct_diagnosis_executed.ipynb
# (PubMedBERT + InfoNCE Knowledge Encoder, proj_dim=256). NOT used by the paper.
# ---------------------------------------------------------------------------
class KnowledgeEncoderStub:  # pragma: no cover
    """Placeholder for KAD-style PubMedBERT/InfoNCE entity embeddings.

    Intentionally unimplemented: the paper uses the symbolic rule engine. Wire up
    only if extending to full neuro-symbolic embedding fusion.
    """

    def __init__(self, *_, **__):
        raise NotImplementedError(
            "KnowledgeEncoderStub: neural entity encoder is a future extension "
            "(see kg.py docstring). The symbolic GuidelineKG is the paper's KG."
        )


if __name__ == "__main__":
    kg = GuidelineKG.load_default()
    for bm, desc in [
        ({"IRF": 1, "SRF": 0, "PED": 0, "HRF": 1}, "IRF only"),
        ({"IRF": 0, "SRF": 1, "PED": 0, "HRF": 1}, "SRF present"),
        ({"IRF": 0, "SRF": 0, "PED": 0, "HRF": 0}, "dry macula"),
        ({"IRF": 0, "SRF": 0, "PED": 1, "HRF": 0}, "PED only"),
    ]:
        print(f"{desc:14s} -> {kg.forward_chain(bm)}")
    print("\nRAG context:\n" + kg.render_rag_context())
