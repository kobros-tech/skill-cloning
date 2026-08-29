"""
strategies.py

Phase 3 baselines + the proposed mechanism, run over the task sequence
addition -> subtraction -> multiplication -> powers (tasks.TASK_ORDER).

Each strategy returns a dict of skill-name -> Skill, plus a training log that
records, for every (task_just_trained, task_evaluated) pair, the MSE right
after that training step. That log is what lets us see forgetting appear
(or not) as later tasks are learned.
"""
from __future__ import annotations
import numpy as np
from skill import TinyMLP, Skill
from tasks import sample_task, TASK_ORDER
import compatibility as comp

N_TRAIN = 200
N_EVAL = 300
EPOCHS = 1500
LR = 0.02
ACC_TOL = 0.5        # a prediction within +-0.5 of the true integer answer counts as "correct"
ACC_TARGET = 0.85     # convergence = training-batch accuracy reaches this level
HIDDEN_DIM = 32


def _eval_all_tasks(net, seed: int) -> dict:
    out = {}
    for t in TASK_ORDER:
        Xe, ye = sample_task(t, N_EVAL, seed=seed + 555)
        out[t] = net.mse(Xe, ye)
    return out


def _train_track_accuracy(net, X, y, epochs, lr, target_acc):
    """Full-batch Adam training; returns (mse_history, steps_to_target_acc_or_epochs)."""
    mse_history = []
    steps = epochs  # default: never reached
    reached = False
    for e in range(epochs):
        net.train_step(X, y, lr=lr, trainable=None)
        mse_history.append(net.mse(X, y))
        if not reached and net.accuracy(X, y, tol=ACC_TOL) >= target_acc:
            steps = e + 1
            reached = True
            break
    return mse_history, steps


def run_shared(seed: int) -> dict:
    """Baseline A: one network, all params trainable, trained sequentially on every task."""
    net = TinyMLP(hidden_dim=HIDDEN_DIM, seed=seed)
    net.reset_optimizer()
    log = []  # rows: {trained_on, evaluated_on, mse, step_in_curriculum}
    convergence = {}
    for step, task in enumerate(TASK_ORDER):
        X, y = sample_task(task, N_TRAIN, seed=seed * 100 + step)
        _, steps = _train_track_accuracy(net, X, y, EPOCHS, LR, ACC_TARGET)
        convergence[task] = steps
        snap = _eval_all_tasks(net, seed=seed)
        for t, m in snap.items():
            log.append({"trained_on": task, "evaluated_on": t, "mse": m, "curriculum_step": step})
    final = _eval_all_tasks(net, seed=seed)
    final_acc = {}
    for t in TASK_ORDER:
        Xe, ye = sample_task(t, N_EVAL, seed=seed + 555)
        final_acc[t] = net.accuracy(Xe, ye)
    return {
        "strategy": "shared_sequential",
        "log": log,
        "convergence_steps": convergence,
        "final_mse": final,
        "final_acc": final_acc,
        "total_params": net.num_params(),
    }


def run_scratch(seed: int) -> dict:
    """Baseline B: independent network per task, always random init. No forgetting by
    construction; used as the retention upper bound and the transfer-benefit lower bound."""
    log = []
    convergence = {}
    nets = {}
    for step, task in enumerate(TASK_ORDER):
        net = TinyMLP(hidden_dim=HIDDEN_DIM, seed=seed * 100 + step)
        X, y = sample_task(task, N_TRAIN, seed=seed * 100 + step)
        _, steps = _train_track_accuracy(net, X, y, EPOCHS, LR, ACC_TARGET)
        convergence[task] = steps
        nets[task] = net
        for t in TASK_ORDER:
            Xe, ye = sample_task(t, N_EVAL, seed=seed + 555)
            m = net.mse(Xe, ye) if t == task else np.nan  # scratch net for `task` was never trained on `t`
            log.append({"trained_on": task, "evaluated_on": t, "mse": m, "curriculum_step": step})
    final = {}
    final_acc = {}
    for t in TASK_ORDER:
        Xe, ye = sample_task(t, N_EVAL, seed=seed + 555)
        final[t] = nets[t].mse(Xe, ye)
        final_acc[t] = nets[t].accuracy(Xe, ye)
    return {
        "strategy": "independent_scratch",
        "log": log,
        "convergence_steps": convergence,
        "final_mse": final,
        "final_acc": final_acc,
        "total_params": sum(n.num_params() for n in nets.values()),
    }


def run_proposed(seed: int) -> dict:
    """The mechanism from Sections 4-6: compatibility-gated reuse / clone / new-skill."""
    skills: dict[str, Skill] = {}
    skill_for_task: dict[str, str] = {}
    log = []
    convergence = {}
    decisions = {}
    for step, task in enumerate(TASK_ORDER):
        decision = comp.decide(skills, task, base_seed=seed * 100 + step)
        decisions[task] = decision
        X, y = sample_task(task, N_TRAIN, seed=seed * 100 + step)

        if decision["action"] == "reuse":
            parent_skill = skills[decision["parent"]]
            parent_skill.tasks_covered.append(task)
            skill_for_task[task] = parent_skill.name
            convergence[task] = 0  # no training needed
        else:
            if decision["action"] == "clone":
                parent_skill = skills[decision["parent"]]
                new_net = parent_skill.net.clone()
                new_net.reset_optimizer()
                new_skill = Skill(task, new_net, origin="clone", parent=parent_skill.name)
            else:  # scratch
                new_net = TinyMLP(hidden_dim=HIDDEN_DIM, seed=seed * 100 + step)
                new_skill = Skill(task, new_net, origin="scratch", parent=None)

            # trains only this skill's own (cloned or fresh) params; the parent
            # object referenced by `skills[decision["parent"]]` is never touched (Section 5)
            _, steps = _train_track_accuracy(new_net, X, y, EPOCHS, LR, ACC_TARGET)
            convergence[task] = steps
            skills[task] = new_skill
            skill_for_task[task] = task

        # snapshot: evaluate every skill currently in the system on every task seen so far
        for t in TASK_ORDER[: step + 1]:
            sk = skills[skill_for_task[t]]
            Xe, ye = sample_task(t, N_EVAL, seed=seed + 555)
            m = sk.net.mse(Xe, ye)
            log.append({"trained_on": task, "evaluated_on": t, "mse": m, "curriculum_step": step})

    final = {}
    final_acc = {}
    for t in TASK_ORDER:
        sk = skills[skill_for_task[t]]
        Xe, ye = sample_task(t, N_EVAL, seed=seed + 555)
        final[t] = sk.net.mse(Xe, ye)
        final_acc[t] = sk.net.accuracy(Xe, ye)
    return {
        "strategy": "clone_and_adapt",
        "log": log,
        "convergence_steps": convergence,
        "final_mse": final,
        "final_acc": final_acc,
        "total_params": sum(s.net.num_params() for s in skills.values()),
        "decisions": {t: {"action": d["action"], "parent": d["parent"], "score": d["score"]}
                      for t, d in decisions.items()},
        "skill_for_task": skill_for_task,
    }
