"""The 12 rule detectors (R2 section 4), weekly features and labels.

Ranking signals: A1 A2 A3 A4 H1 H2 G1 G2 E1 - a student is eligible for
the weekly list only if at least one of these fired.
Section signal: S1 - routed to the academic head, suppresses G1 for that
class and subject.
Annotations: T1 (timetable link) and F1 (fee delay, restricted roles) -
never used for ranking.

Every fired signal carries a fact dictionary; reason text is rendered from
those facts only.
"""
from __future__ import annotations

from dataclasses import dataclass

import warnings

import numpy as np

from . import config as C
from .features import FEATURES, SchoolArrays, Windows, rate

T = C.T


@dataclass
class WeekResults:
    feats: np.ndarray  # (W, n, F)
    fired: dict  # signal -> (W, n) bool
    facts: dict  # (t, i) -> {signal: facts}
    s1_events: list  # dicts
    dq_incomplete: set  # {(class, t)}
    labels: np.ndarray  # (W, n) float, nan when the horizon runs past the year


def _pct(x: float) -> int:
    return int(round(100 * float(x)))


def _streak_runs(flags: np.ndarray) -> np.ndarray:
    """Length of the current consecutive run ending on each day."""
    run = np.zeros(flags.shape, np.int16)
    cur = np.zeros(flags.shape[0], np.int16)
    for d in range(flags.shape[1]):
        cur = np.where(flags[:, d], cur + 1, 0)
        run[:, d] = cur
    return run


def section_slips(A: SchoolArrays) -> list[dict]:
    """S1 events: a class-subject whose median fell between two consecutive assessments."""
    events = []
    for c in sorted(set(A.klass)):
        members = np.where(A.klass == c)[0]
        for s, subject in enumerate(C.SUBJECTS):
            m = A.marks[members, s, :]
            for a in range(1, m.shape[1]):
                diff = m[:, a] - m[:, a - 1]
                med_change = float(np.nanmedian(m[:, a]) - np.nanmedian(m[:, a - 1]))
                dropped = int(np.sum(diff <= -T["S1_median_drop"]))
                if med_change <= -T["S1_median_drop"] and dropped / len(members) >= T["S1_share"]:
                    events.append({
                        "class": c, "subject": subject, "subject_idx": s, "assessment": a,
                        "week0": int(A.assess_week[s, a]),
                        "from": int(round(float(np.nanmedian(m[:, a - 1])))),
                        "to": int(round(float(np.nanmedian(m[:, a])))),
                        "dropped": dropped, "size": len(members),
                    })
    return events


def compute(A: SchoolArrays) -> WeekResults:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _compute(A)


def _compute(A: SchoolArrays) -> WeekResults:
    n, nW, nS = A.n, C.N_WEEKS, len(C.SUBJECTS)
    W_marked = Windows(A.weekly(A.marked))
    W_present = Windows(A.weekly(A.present))
    W_absent = Windows(A.weekly(A.absent_any))
    W_unexc = Windows(A.weekly(A.unexcused))
    W_late_arr = Windows(A.weekly(A.late_arrival))
    W_lms = Windows(A.weekly(A.lms_active))
    W_due = Windows(A.due.sum(axis=2))
    W_miss = Windows(A.missing.sum(axis=2))
    W_late = Windows(A.late_sub.sum(axis=2))
    W_due_s = Windows(A.due)
    W_miss_s = Windows(A.missing)
    # absences by weekday: (n, W, 5)
    abs_wd = A.absent_any.reshape(n, nW, 5)
    W_abs_wd = Windows(abs_wd)
    unexc_run = _streak_runs(A.unexcused)
    last_active = np.where(A.lms_active, np.arange(C.N_DAYS)[None, :], -1)
    last_active = np.maximum.accumulate(last_active, axis=1)

    s1 = section_slips(A)
    s1_by_cs: dict = {}
    for e in s1:
        s1_by_cs.setdefault((e["class"], e["subject_idx"]), []).append(e)

    # data-quality gate per class-week
    classes = sorted(set(A.klass))
    weekly_marked = A.weekly(A.marked)
    dq = set()
    for c in classes:
        members = A.klass == c
        share = weekly_marked[members].sum(axis=0) / (members.sum() * C.DAYS_PER_WEEK)
        for t in np.where(share < T["DQ_min_marked"])[0]:
            dq.add((c, int(t)))

    feats = np.full((nW, n, len(FEATURES)), np.nan)
    fired = {s: np.zeros((nW, n), bool) for s in C.RANKING_SIGNALS + C.ANNOTATION_SIGNALS}
    facts: dict = {}
    run_start = {"A1": np.full(n, -1), "G1": np.full(n, -1)}
    earlier_flags = np.zeros(n, int)

    for t in range(nW):
        ts, te = C.term_of(t)
        if t == ts:
            earlier_flags[:] = 0
        end_day = t * 5 + 4
        has4 = t + 1 >= C.MIN_WEEKS_FOR_SIGNALS
        has12 = t + 1 >= C.MIN_WEEKS_FOR_BASELINE

        att_w4 = rate(W_present.sum(t, 4), W_marked.sum(t, 4))
        att_b8 = rate(W_present.sum(t, 8, gap=4), W_marked.sum(t, 8, gap=4))
        ttd_len = t - ts + 1
        ttd_p, ttd_m = W_present.sum(t, ttd_len), W_marked.sum(t, ttd_len)
        att_ttd = rate(ttd_p, ttd_m)
        remaining = (te - t - 1) * C.DAYS_PER_WEEK
        proj = rate(ttd_p + np.nan_to_num(att_w4) * remaining, ttd_m + remaining)
        abs6 = W_abs_wd.sum(t, T["A3_weeks"])  # (n, 5)
        abs6_tot = abs6.sum(axis=1)
        wd_max = abs6.max(axis=1)
        wd_arg = abs6.argmax(axis=1)
        wd_share = rate(wd_max, abs6_tot)
        lo = max(0, end_day - T["A4_lookback_days"] + 1)
        streak = unexc_run[:, lo:end_day + 1].max(axis=1)

        due4, miss4 = W_due.sum(t, 4), W_miss.sum(t, 4)
        due8, miss8 = W_due.sum(t, 8, gap=4), W_miss.sum(t, 8, gap=4)
        miss_r4, miss_r8 = rate(miss4, due4), rate(miss8, due8)
        sub4, sub8 = due4 - miss4, due8 - miss8
        late_r4, late_r8 = rate(W_late.sum(t, 4), sub4), rate(W_late.sum(t, 8, gap=4), sub8)
        miss_s4 = W_miss_s.sum(t, 4)  # (n, S)
        due_s4 = W_due_s.sum(t, 4)
        max_s_miss = miss_s4.max(axis=1)
        max_s_arg = miss_s4.argmax(axis=1)
        days_since_sub = np.where(A.last_sub_day[:, t] >= 0, end_day - A.last_sub_day[:, t], 60)

        lms4, lms8 = W_lms.sum(t, 4), W_lms.sum(t, 8, gap=4)
        lms_base4 = lms8 / 2.0
        lms_ratio = rate(lms4, lms_base4)
        days_since_lms = np.where(last_active[:, end_day] >= 0, end_day - last_active[:, end_day], 60)

        # assessments available by the end of week t
        k_avail = (A.assess_week <= t).sum(axis=1)  # (S,)
        last_mark = np.full((n, nS), np.nan)
        last3 = np.full((n, nS), np.nan)
        drop = np.full((n, nS), np.nan)
        earlier_mean = np.full((n, nS), np.nan)
        last2_mean = np.full((n, nS), np.nan)
        trend = np.full((n, nS), np.nan)
        for s in range(nS):
            k = k_avail[s]
            if k >= 1:
                last_mark[:, s] = A.marks[:, s, k - 1]
            if k >= 2:
                trend[:, s] = A.marks[:, s, k - 1] - A.marks[:, s, k - 2]
            if k >= 3:
                last3[:, s] = A.marks[:, s, k - 3:k].mean(axis=1)
            if k >= 2 + T["G1_min_earlier"]:
                last2_mean[:, s] = A.marks[:, s, k - 2:k].mean(axis=1)
                earlier_mean[:, s] = A.marks[:, s, :k - 2].mean(axis=1)
                drop[:, s] = last2_mean[:, s] - earlier_mean[:, s]
        # G1 suppression by S1 (section slip inside the last-two window)
        suppress = np.zeros((n, nS), bool)
        slip_now = np.zeros(n, bool)
        for (c, s), evs in s1_by_cs.items():
            k = k_avail[s]
            if any(e["assessment"] in (k - 2, k - 1) for e in evs if e["week0"] <= t):
                members = A.klass == c
                suppress[members, s] = True
                slip_now |= members
        core = [C.SUBJECTS.index(x) for x in C.CORE_SUBJECTS]
        with np.errstate(all="ignore"):
            score_mean = np.nanmean(np.where(np.arange(A.marks.shape[2])[None, None, :] < k_avail[None, :, None], A.marks, np.nan), axis=(1, 2))
            z = np.full(n, np.nan)
            for c in classes:
                mm = A.klass == c
                lm = last_mark[mm]
                z[mm] = np.nanmean((lm - np.nanmean(lm, axis=0)) / (np.nanstd(lm, axis=0) + 1e-6), axis=1)
            min_last3 = np.nanmin(last3, axis=1) if np.isfinite(last3).any() else np.full(n, np.nan)
            max_drop = np.nanmin(drop, axis=1) if np.isfinite(drop).any() else np.full(n, np.nan)
            core_min = np.nanmin(last_mark[:, core], axis=1) if np.isfinite(last_mark[:, core]).any() else np.full(n, np.nan)
            mean_trend = np.nanmean(trend, axis=1) if np.isfinite(trend).any() else np.full(n, np.nan)
            mean_last = np.nanmean(last_mark, axis=1) if np.isfinite(last_mark).any() else np.full(n, np.nan)

        future = A.assess_week[A.assess_week > t]
        weeks_to_exam = (future.min() - t) if future.size else 10
        tt_recent = np.array([any(0 <= t - w < T["T1_weeks"] for _, w in A.tt_changes.get(c, [])) for c in A.klass])

        # ---------------- rules ----------------
        f = {}
        f["A1"] = has12 & (100 * (att_w4 - att_b8) <= -T["A1_drop_pts"]) & (att_w4 < T["A1_max_w4"])
        f["A2"] = (ttd_len >= 4) & ((att_ttd < T["A2_ttd"]) | (proj < T["A2_projection"]))
        f["A3"] = has4 & (abs6_tot >= T["A3_min_absences"]) & (wd_share >= T["A3_share"])
        f["A4"] = streak >= T["A4_streak"]
        h1_subj = max_s_miss >= T["H1_subject_missing"]
        h1_all = (due4 >= T["H1_min_due"]) & (miss_r4 >= T["H1_overall_rate"])
        f["H1"] = has4 & (h1_subj | h1_all)
        f["H2"] = has12 & (late_r4 >= T["H2_rate"]) & (100 * (late_r4 - late_r8) >= T["H2_rise_pts"])
        g1_s = (drop <= -T["G1_drop_pts"]) & ~suppress
        f["G1"] = g1_s.any(axis=1)
        g2_s = last3 < C.PASS_MARK + T["G2_margin_pts"]
        f["G2"] = g2_s.any(axis=1)
        f["E1"] = has12 & (lms_base4 >= T["E1_min_base_days_per_4w"]) & (lms_ratio <= T["E1_ratio"])
        for key in f:
            f[key] = np.nan_to_num(f[key]).astype(bool)

        # T1: onset of A1 / G1 within 3 weeks after a timetable change
        for key in ("A1", "G1"):
            rs = run_start[key]
            rs[:] = np.where(f[key], np.where(rs >= 0, rs, t), -1)
        t1 = np.zeros(n, bool)
        t1_week = np.full(n, -1)
        for i in np.where(f["A1"] | f["G1"])[0]:
            for s, w in A.tt_changes.get(A.klass[i], []):
                for key in ("A1", "G1"):
                    if not f[key][i]:
                        continue
                    if key == "G1" and not g1_s[i, s]:
                        continue
                    if 0 <= run_start[key][i] - w <= T["T1_weeks"]:
                        t1[i] = True
                        t1_week[i] = w
        f["T1"] = t1
        # F1: fee invoice overdue more than 30 calendar days (context only)
        end_date = C.day_date(end_day)
        unpaid = (A.fee_due <= end_day) & ((A.fee_paid < 0) | (A.fee_paid > end_day))
        overdue_days = np.zeros(n, int)
        for i, j in zip(*np.where(unpaid)):
            overdue_days[i] = max(overdue_days[i], (end_date - C.day_date(int(A.fee_due[i, j]))).days)
        f["F1"] = overdue_days > T["F1_days"]

        for key in fired:
            fired[key][t] = f[key]

        # ---------------- features ----------------
        week_flags = np.zeros(n, bool)
        for key in C.RANKING_SIGNALS:
            week_flags |= f[key]
        cols = [
            att_w4, att_b8, att_w4 - att_b8, att_ttd, proj, wd_share, abs6_tot, streak,
            W_late_arr.sum(t, 4), W_unexc.sum(t, 4),
            miss_r4, miss_r8, miss_r4 - miss_r8, late_r4, late_r8, late_r4 - late_r8, max_s_miss, days_since_sub,
            score_mean, mean_last, min_last3, max_drop, min_last3 - C.PASS_MARK, z,
            np.full(n, k_avail.mean()), core_min, mean_trend,
            lms4, lms8, lms_ratio, W_lms.sum(t, 1), days_since_lms,
            A.grade, np.where(A.new_student, t + 1, t + 1 + 52 * A.years_enrolled), A.new_student,
            np.full(n, t - ts + 1), np.full(n, weeks_to_exam), tt_recent, slip_now, earlier_flags.copy(),
            np.zeros(n),
        ]
        feats[t] = np.column_stack([np.asarray(c, float) for c in cols])
        earlier_flags += week_flags

        # ---------------- facts for every fired signal ----------------
        any_fired = week_flags | f["T1"] | f["F1"]
        for i in np.where(any_fired)[0]:
            d = {}
            if f["A1"][i]:
                d["A1"] = {"from": _pct(att_b8[i]), "to": _pct(att_w4[i]), "weeks": 4}
            if f["A2"][i]:
                if proj[i] < T["A2_projection"]:
                    d["A2"] = {"projection": _pct(proj[i]), "minimum": _pct(C.MIN_ATTENDANCE), "mode": "projection"}
                else:
                    d["A2"] = {"ttd": _pct(att_ttd[i]), "threshold": _pct(T["A2_ttd"]), "mode": "ttd"}
            if f["A3"][i]:
                d["A3"] = {"k": int(wd_max[i]), "absences": int(abs6_tot[i]), "weekday": C.WEEKDAYS[wd_arg[i]], "weeks": T["A3_weeks"]}
            if f["A4"][i]:
                d["A4"] = {"days": int(streak[i])}
            if f["H1"][i]:
                if h1_subj[i]:
                    s = int(max_s_arg[i])
                    d["H1"] = {"mode": "subject", "subject": C.SUBJECTS[s], "missing": int(miss_s4[i, s]), "due": int(due_s4[i, s]), "weeks": 4}
                else:
                    d["H1"] = {"mode": "overall", "rate": _pct(miss_r4[i]), "missing": int(miss4[i]), "due": int(due4[i]), "weeks": 4}
            if f["H2"][i]:
                d["H2"] = {"from": _pct(late_r8[i]), "to": _pct(late_r4[i])}
            if f["G1"][i]:
                s = int(np.nanargmin(np.where(g1_s[i], drop[i], np.nan)))
                d["G1"] = {"subject": C.SUBJECTS[s], "from": int(round(earlier_mean[i, s])), "to": int(round(last2_mean[i, s]))}
            if f["G2"][i]:
                s = int(np.nanargmin(np.where(g2_s[i], last3[i], np.nan)))
                avg = int(round(last3[i, s]))
                d["G2"] = {"subject": C.SUBJECTS[s], "avg": avg, "pass_mark": int(C.PASS_MARK), "margin": avg - int(C.PASS_MARK)}
            if f["E1"][i]:
                d["E1"] = {"from": int(round(lms_base4[i])), "to": int(lms4[i])}
            if f["T1"][i]:
                d["T1"] = {"change_week": int(t1_week[i]) + 1, "weeks": T["T1_weeks"]}
            if f["F1"][i]:
                d["F1"] = {"days": int(overdue_days[i])}
            facts[(t, int(i))] = d

    labels = make_labels(A)
    return WeekResults(feats=feats, fired=fired, facts=facts, s1_events=s1, dq_incomplete=dq, labels=labels)


def make_labels(A: SchoolArrays) -> np.ndarray:
    """Decline within the next 6 weeks (R2 section 4.1)."""
    n, nW, H = A.n, C.N_WEEKS, C.LABEL_HORIZON
    lab = np.full((nW, n), np.nan)
    wm, wp = Windows(A.weekly(A.marked)), Windows(A.weekly(A.present))
    wd, wmiss = Windows(A.due.sum(axis=2)), Windows(A.missing.sum(axis=2))
    core = [C.SUBJECTS.index(x) for x in C.CORE_SUBJECTS]
    for t in range(nW - H):
        hi = t + H
        att = rate(wp.sum(hi, H), wm.sum(hi, H))
        miss = rate(wmiss.sum(hi, H), wd.sum(hi, H))
        fail = np.zeros(n, bool)
        for s in core:
            sel = (A.assess_week[s] > t) & (A.assess_week[s] <= hi)
            if sel.any():
                fail |= np.nanmin(A.marks[:, s, sel], axis=1) < C.PASS_MARK
        lab[t] = ((att < C.LABEL_ATTENDANCE) | (miss >= C.LABEL_MISSING) | fail).astype(float)
    return lab


def severity(fired: dict, t: int) -> np.ndarray:
    sev = np.zeros(next(iter(fired.values())).shape[1])
    for s in C.RANKING_SIGNALS:
        sev += C.WEIGHTS[s] * fired[s][t]
    return np.minimum(sev, 1.0)


def eligible(fired: dict, t: int) -> np.ndarray:
    e = np.zeros(next(iter(fired.values())).shape[1], bool)
    for s in C.RANKING_SIGNALS:
        e |= fired[s][t]
    return e
