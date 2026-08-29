"""
compatibility.py

Section 3 leaves P(T | s_i) intentionally undefined. Design choice made here:

    P(T | s_i) = exp( -MSE_i(T) / SCALE )

i.e. evaluate skill i (no gradient update) on a probe batch from task T, and
squash the raw MSE into (0, 1] with an exponential. SCALE is a fixed constant
chosen empirically from the target magnitudes of the arithmetic tasks (targets
range roughly 0-90), so that "solves it well" -> score near 1 and "unrelated" ->
score near 0.

Decision rule (Sections 4-6), thresholds chosen for the prototype:
    score >= TAU_SOLVE  -> reuse existing skill, no training at all
    score >= TAU_CLONE  -> clone the best skill, train only the clone
    else                -> create a new skill from random init
"""
from __future__ import annotations
import numpy as np

SCALE = 60.0
TAU_SOLVE = 0.90
TAU_CLONE = 0.15
PROBE_N = 64
PROBE_SEED_OFFSET = 10_000  # keep probe samples disjoint-ish from training seeds


def compatibility_score(skill_net, task: str, base_seed: int) -> float:
    from tasks import sample_task
    X, y = sample_task(task, PROBE_N, seed=base_seed + PROBE_SEED_OFFSET)
    mse = skill_net.mse(X, y)
    return float(np.exp(-mse / SCALE))


def rank_skills(skills: dict, task: str, base_seed: int) -> list[tuple[str, float]]:
    """skills: {name: Skill}. Returns [(name, score), ...] sorted descending."""
    scored = [(name, compatibility_score(s.net, task, base_seed)) for name, s in skills.items()]
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored


def decide(skills: dict, task: str, base_seed: int):
    """
    Returns a dict describing the decision:
      {"action": "reuse"|"clone"|"scratch", "parent": name_or_None, "score": float,
       "ranking": [(name, score), ...]}
    """
    if not skills:
        return {"action": "scratch", "parent": None, "score": 0.0, "ranking": []}
    ranking = rank_skills(skills, task, base_seed)
    best_name, best_score = ranking[0]
    if best_score >= TAU_SOLVE:
        return {"action": "reuse", "parent": best_name, "score": best_score, "ranking": ranking}
    if best_score >= TAU_CLONE:
        return {"action": "clone", "parent": best_name, "score": best_score, "ranking": ranking}
    return {"action": "scratch", "parent": None, "score": best_score, "ranking": ranking}
