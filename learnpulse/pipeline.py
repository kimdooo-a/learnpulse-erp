"""Weekly batch and evaluation.

`build()` runs the whole chain once for the synthetic school year:
generate -> ERP tables -> connector -> features -> 12 detectors -> model ->
priority -> reasons -> stored results + evaluation report.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from . import config as C
from . import generate as G
from . import store
from .features import FEATURES, SIGNAL_FEATURES, build_arrays
from .model import train
from .reasons import reason_en, verify
from .signals import compute, eligible, severity


def rank_lists(klass: np.ndarray, elig: np.ndarray, priority: np.ndarray,
               excluded: np.ndarray | None = None, size: int = C.LIST_SIZE) -> np.ndarray:
    """Indices of the top `size` eligible students per class, by priority."""
    ok = elig.copy()
    if excluded is not None:
        ok &= ~excluded
    out = []
    for c in np.unique(klass):
        idx = np.where((klass == c) & ok)[0]
        if idx.size:
            order = idx[np.argsort(-priority[idx], kind="stable")]
            out.extend(order[:size].tolist())
    return np.array(sorted(out), int)


def reason_order(signals: list[str], contrib_row: np.ndarray | None) -> list[str]:
    """Order fired ranking signals by the attribution of their features (top 3 shown)."""
    if contrib_row is None:
        return sorted(signals, key=lambda s: -C.WEIGHTS[s])
    fi = {f: k for k, f in enumerate(FEATURES)}
    score = {s: float(np.sum(np.abs(contrib_row[[fi[f] for f in SIGNAL_FEATURES[s]]]))) for s in signals}
    return sorted(signals, key=lambda s: (-score[s], -C.WEIGHTS[s]))


def _dq_mask(klass: np.ndarray, dq: set, t: int) -> np.ndarray:
    bad = {c for c, w in dq if w == t}
    return np.isin(klass, list(bad))


def evaluate(A, R, prob: np.ndarray, truth: pd.DataFrame) -> dict:
    """R2 section 8.2 metrics on the synthetic dataset."""
    n = A.n
    t0, t1 = C.TEST_WEEKS[0] - 1, C.TEST_WEEKS[1] - 1
    ts = range(t0, t1 + 1)
    on_traj = np.zeros(n, bool)
    onset = np.full(n, 10_000)
    ttype = np.array([""] * n, dtype=object)
    sid_idx = {s: i for i, s in enumerate(A.sid)}
    for r in truth.itertuples():
        i = sid_idx[r.student_id]
        on_traj[i] = True
        onset[i] = r.onset_week
        ttype[i] = r.trajectory

    # ---- model quality: PR-AUC on test weeks vs two baselines ----
    y, p_model, p_rules, p_att = [], [], [], []
    att_col = FEATURES.index("att_ttd")
    for t in ts:
        keep = ~np.isnan(R.labels[t])
        y.append(R.labels[t][keep])
        p_model.append(prob[t][keep])
        p_rules.append(severity(R.fired, t)[keep])
        p_att.append((R.feats[t][:, att_col] < C.MIN_ATTENDANCE).astype(float)[keep])
    y = np.concatenate(y)
    res = {"test_rows": len(y), "test_positive_rate": round(float(y.mean()), 4)}
    for name, p in (("model", p_model), ("rules_only", p_rules), ("attendance_below_75", p_att)):
        p = np.concatenate(p)
        res[f"pr_auc_{name}"] = round(float(average_precision_score(y, p)), 4)
        res[f"roc_auc_{name}"] = round(float(roc_auc_score(y, p)), 4)

    # ---- weekly list precision on test weeks ----
    def list_stats(prio_fn, weeks):
        listed = hit_traj = hit_label = 0
        for t in weeks:
            el = eligible(R.fired, t)
            L = rank_lists(A.klass, el, prio_fn(t), excluded=_dq_mask(A.klass, R.dq_incomplete, t))
            listed += len(L)
            hit_traj += int(np.sum(on_traj[L] & (onset[L] <= t + 1)))
            lab = R.labels[t][L]
            hit_label += int(np.nansum(lab))
        return listed, hit_traj, hit_label

    full = lambda t: C.MIX_MODEL * prob[t] + C.MIX_RULES * severity(R.fired, t)
    rules = lambda t: severity(R.fired, t)
    for name, fn in (("priority", full), ("rules_only", rules)):
        listed, ht, hl = list_stats(fn, ts)
        res[f"list_{name}_listed"] = listed
        res[f"list_{name}_precision_trajectory"] = round(ht / max(listed, 1), 4)
        res[f"list_{name}_precision_label"] = round(hl / max(listed, 1), 4)

    # ---- recall within 4 weeks of onset and lead time (rules-only list, whole year: no training involved) ----
    first_listed = np.full(n, -1)
    first_eligible = np.full(n, -1)
    for t in range(C.N_WEEKS):
        el = eligible(R.fired, t)
        L = rank_lists(A.klass, el, rules(t), excluded=_dq_mask(A.klass, R.dq_incomplete, t))
        new = np.zeros(n, bool)
        new[L] = True
        win = (t + 1 >= onset)
        first_listed = np.where((first_listed < 0) & new & win, t + 1, first_listed)
        first_eligible = np.where((first_eligible < 0) & el & win, t + 1, first_eligible)
    measurable = on_traj & (onset + 3 <= C.N_WEEKS) & (onset >= C.MIN_WEEKS_FOR_SIGNALS + 1)
    within_list = measurable & (first_listed > 0) & (first_listed <= onset + 3)
    within_elig = measurable & (first_eligible > 0) & (first_eligible <= onset + 3)
    res["recall_trajectories_measurable"] = int(measurable.sum())
    res["recall_listed_within_4w"] = round(float(within_list.sum() / measurable.sum()), 4)
    res["recall_signal_within_4w"] = round(float(within_elig.sum() / measurable.sum()), 4)
    by_type = {}
    for k in G.TRAJECTORIES:
        m = measurable & (ttype == k)
        by_type[k] = {"n": int(m.sum()), "listed_within_4w": round(float((within_list & m).sum() / max(m.sum(), 1)), 4)}
    res["recall_by_trajectory"] = by_type
    # lead time vs a term-end review (week 20 / week 40)
    leads = []
    for i in np.where(on_traj & (first_listed > 0))[0]:
        term_end = 20 if onset[i] <= 20 else 40
        if first_listed[i] <= term_end:
            leads.append(term_end - first_listed[i])
    res["lead_time_weeks_median"] = float(np.median(leads)) if leads else None
    res["lead_time_students"] = len(leads)

    # ---- S1 vs the two injected section slips ----
    res["section_alerts"] = len(R.s1_events)
    return res


def build(db: Path = store.DEFAULT_DB, seed: int = C.SEED, report_dir: Path = Path("reports")) -> dict:
    timings = {}
    t = time.perf_counter()
    data = G.generate(seed)
    timings["generate_s"] = time.perf_counter() - t

    t = time.perf_counter()
    if db.exists():
        db.unlink()
    store.write_tables(db, data)
    timings["load_erp_tables_s"] = time.perf_counter() - t

    t = time.perf_counter()
    erp = store.read_erp(db)  # read-only connector
    A = build_arrays(erp)
    timings["connector_and_arrays_s"] = time.perf_counter() - t

    t = time.perf_counter()
    R = compute(A)
    timings["features_and_12_signals_40_weeks_s"] = time.perf_counter() - t

    t = time.perf_counter()
    model, model_info = train(R.feats, R.labels, seed)
    timings["train_model_s"] = time.perf_counter() - t

    t = time.perf_counter()
    nW, n = C.N_WEEKS, A.n
    prob = np.zeros((nW, n))
    contrib = np.zeros((nW, n, len(FEATURES)))
    for w in range(nW):
        prob[w] = model.predict(R.feats[w])
        contrib[w] = model.contributions(R.feats[w])
    timings["score_40_weeks_s"] = time.perf_counter() - t

    # one weekly batch for one week (what runs every Sunday night), timed separately
    t = time.perf_counter()
    week_t = C.TEST_WEEKS[0] - 1
    _ = model.predict(R.feats[week_t]), model.contributions(R.feats[week_t])
    _ = rank_lists(A.klass, eligible(R.fired, week_t), C.MIX_MODEL * prob[week_t] + C.MIX_RULES * severity(R.fired, week_t))
    timings["one_week_score_and_rank_s"] = time.perf_counter() - t

    # ---------------- persist results ----------------
    t = time.perf_counter()
    score_rows, flag_rows, reason_rows = [], [], []
    texts_checked = texts_ok = 0
    for w in range(nW):
        sev = severity(R.fired, w)
        el = eligible(R.fired, w)
        pr = C.MIX_MODEL * prob[w] + C.MIX_RULES * sev
        lab = R.labels[w]
        for i in range(n):
            score_rows.append((w + 1, A.sid[i], A.klass[i], float(prob[w, i]), float(sev[i]), float(pr[i]), int(el[i]),
                               None if np.isnan(lab[i]) else int(lab[i])))
        for (tw, i), d in R.facts.items():
            if tw != w:
                continue
            for s, f in d.items():
                text = reason_en(s, f)
                texts_checked += 1
                texts_ok += verify(text, f)
                flag_rows.append((w + 1, A.sid[i], s, json.dumps(f), text))
            ranking = [s for s in d if s in C.RANKING_SIGNALS]
            if ranking:
                for rank, s in enumerate(reason_order(ranking, contrib[w, i])):
                    reason_rows.append((w + 1, A.sid[i], rank + 1, s))
    snap = pd.DataFrame(R.feats.reshape(nW * n, -1), columns=FEATURES)
    snap.insert(0, "week", np.repeat(np.arange(1, nW + 1), n))
    snap.insert(1, "student_id", np.tile(A.sid, nW))
    with sqlite3.connect(db) as con:
        pd.DataFrame(score_rows, columns=["week", "student_id", "class", "model_prob", "severity", "priority", "eligible", "label"]).to_sql("weekly_scores", con, index=False, if_exists="replace")
        pd.DataFrame(flag_rows, columns=["week", "student_id", "signal", "facts", "text"]).to_sql("flags", con, index=False, if_exists="replace")
        pd.DataFrame(reason_rows, columns=["week", "student_id", "rank", "signal"]).to_sql("reason_order", con, index=False, if_exists="replace")
        snap.to_sql("feature_snapshots", con, index=False, if_exists="replace", chunksize=20_000)
        s1 = pd.DataFrame([{k: v for k, v in e.items() if k != "subject_idx"} for e in R.s1_events])
        if len(s1):
            s1["week"] = s1.pop("week0") + 1
        s1.to_sql("section_alerts", con, index=False, if_exists="replace")
        pd.DataFrame(sorted((c, w + 1) for c, w in R.dq_incomplete), columns=["class", "week"]).to_sql("data_quality", con, index=False, if_exists="replace")
        for idx in ("CREATE INDEX IF NOT EXISTS ix_scores ON weekly_scores(week, class)",
                    "CREATE INDEX IF NOT EXISTS ix_flags ON flags(week, student_id)",
                    "CREATE INDEX IF NOT EXISTS ix_reason ON reason_order(week, student_id)",
                    "CREATE INDEX IF NOT EXISTS ix_snap ON feature_snapshots(week, student_id)"):
            con.execute(idx)
    timings["persist_results_s"] = time.perf_counter() - t

    metrics = evaluate(A, R, prob, data["truth_trajectories"])
    truth_s = data["truth_sections"]
    s1_hit = [f"{r['class']} {r['subject']} (week {r['week0'] + 1})" for r in R.s1_events]
    report = {
        "dataset": {
            "synthetic": True, "seed": seed,
            "students": int(A.n), "classes": len(set(A.klass)), "weeks": C.N_WEEKS, "subjects": len(C.SUBJECTS),
            "rows": {k: len(data[k]) for k in G.ERP_TABLES},
            "erp_rows_total": int(sum(len(data[k]) for k in G.ERP_TABLES)),
            "trajectory_students": len(data["truth_trajectories"]),
            "trajectory_counts": data["truth_trajectories"]["trajectory"].value_counts().to_dict(),
            "injected_section_slips": [f"{r['class']} {r['subject']} (teacher change week {r['change_week']})" for _, r in truth_s.iterrows()],
            "injected_late_marking": [f"{r['class']} week {r['week']}" for _, r in data["truth_data_quality"].iterrows()],
        },
        "signals_fired_total": {s: int(R.fired[s].sum()) for s in R.fired},
        "section_alerts_s1": s1_hit,
        "data_quality_gate": [f"{c} week {w + 1}" for c, w in sorted(R.dq_incomplete)],
        "model": model_info,
        "metrics": metrics,
        "text_fidelity": {"reason_texts_checked": texts_checked, "numbers_verified": texts_ok,
                          "share": round(texts_ok / max(texts_checked, 1), 4)},
        "timings_seconds": {k: round(v, 3) for k, v in timings.items()},
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (report_dir / "evaluation.md").write_text(render_report(report), encoding="utf-8")
    return report


def render_report(r: dict) -> str:
    m = r["metrics"]
    d = r["dataset"]
    lines = [
        "# LearnPulse - evaluation on the synthetic dataset",
        "",
        "All numbers below are produced by `python -m learnpulse build` on a **synthetic** school "
        f"(seed {d['seed']}). They show that the pipeline works end to end; they are not results from a real school.",
        "",
        "## Dataset",
        f"- {d['students']:,} students, {d['classes']} classes, {d['weeks']} weeks, {d['subjects']} subjects",
        f"- ERP rows: {d['erp_rows_total']:,} ({', '.join(f'{k} {v:,}' for k, v in d['rows'].items())})",
        f"- Injected trajectories: {d['trajectory_students']} students ({', '.join(f'{k} {v}' for k, v in d['trajectory_counts'].items())})",
        f"- Injected section slips: {'; '.join(d['injected_section_slips'])}",
        f"- Injected late attendance marking: {'; '.join(d['injected_late_marking'])}",
        "",
        "## Model (time-ordered split)",
        f"- Backend: {r['model']['backend']}, {r['model']['features']} features; fit weeks {r['model']['fit_weeks']}, "
        f"calibration weeks {r['model']['calibration_weeks']}, test weeks {r['model']['test_weeks']}",
        f"- Test rows {m['test_rows']:,}, positive rate {m['test_positive_rate']}",
        "",
        "| Score | PR-AUC | ROC-AUC |",
        "|---|---|---|",
        f"| LearnPulse model | {m['pr_auc_model']} | {m['roc_auc_model']} |",
        f"| Rules only (severity) | {m['pr_auc_rules_only']} | {m['roc_auc_rules_only']} |",
        f"| Attendance below 75% | {m['pr_auc_attendance_below_75']} | {m['roc_auc_attendance_below_75']} |",
        "",
        "## Weekly list (top 5 per class, test weeks)",
        "",
        "| Ordering | Students listed | On an injected trajectory | Declined within 6 weeks |",
        "|---|---|---|---|",
        f"| Priority = 0.6 model + 0.4 rules | {m['list_priority_listed']} | {m['list_priority_precision_trajectory']} | {m['list_priority_precision_label']} |",
        f"| Rules only | {m['list_rules_only_listed']} | {m['list_rules_only_precision_trajectory']} | {m['list_rules_only_precision_label']} |",
        "",
        "## Recall and lead time (rules-only list, whole year)",
        f"- Trajectories measurable: {m['recall_trajectories_measurable']}",
        f"- Listed within 4 weeks of onset: {m['recall_listed_within_4w']}",
        f"- A ranking signal fired within 4 weeks of onset: {m['recall_signal_within_4w']}",
        "- By trajectory: " + "; ".join(f"{k} {v['listed_within_4w']} (n={v['n']})" for k, v in m['recall_by_trajectory'].items()),
        f"- Median lead time before the term-end review: {m['lead_time_weeks_median']} weeks ({m['lead_time_students']} students)",
        "",
        "## Section alerts (S1) and data-quality gate",
        f"- S1 alerts: {'; '.join(r['section_alerts_s1']) or 'none'}",
        f"- Classes held back as 'data incomplete': {'; '.join(r['data_quality_gate']) or 'none'}",
        "",
        "## Text fidelity",
        f"- Reason texts checked: {r['text_fidelity']['reason_texts_checked']:,}; every number found in the fact list: "
        f"{r['text_fidelity']['numbers_verified']:,} ({r['text_fidelity']['share']:.0%})",
        "",
        "## Runtime on the build machine (seconds)",
        "",
        "| Step | Seconds |",
        "|---|---|",
    ]
    lines += [f"| {k.removesuffix('_s').replace('_', ' ')} | {v} |" for k, v in r["timings_seconds"].items()]
    return "\n".join(lines) + "\n"
