"""Canonical ERP tables -> dense arrays -> weekly snapshot features.

All window arithmetic is done on (student x week) arrays with cumulative
sums, so one snapshot of 1,200 students costs a few milliseconds.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C


@dataclass
class SchoolArrays:
    sid: np.ndarray
    names: np.ndarray
    klass: np.ndarray
    grade: np.ndarray
    lang: np.ndarray
    new_student: np.ndarray
    years_enrolled: np.ndarray
    marked: np.ndarray  # (n, D) bool
    present: np.ndarray  # present or late
    absent_any: np.ndarray  # absent or excused
    unexcused: np.ndarray  # absent without a recorded reason
    late_arrival: np.ndarray
    due: np.ndarray  # (n, W, S) assignments due
    missing: np.ndarray  # (n, W, S)
    late_sub: np.ndarray  # (n, W, S) submitted late
    last_sub_day: np.ndarray  # (n, W) last submission school-day by end of week, -1 none
    lms_active: np.ndarray  # (n, D)
    marks: np.ndarray  # (n, S, A)
    assess_week: np.ndarray  # (S, A) 0-based week
    tt_changes: dict = field(default_factory=dict)  # class -> list[(subject_idx, week0)]
    fee_due: np.ndarray | None = None  # (n, k) school day
    fee_paid: np.ndarray | None = None  # (n, k) school day, -1 unpaid

    @property
    def n(self) -> int:
        return len(self.sid)

    def weekly(self, x: np.ndarray) -> np.ndarray:
        return x.reshape(self.n, C.N_WEEKS, C.DAYS_PER_WEEK).sum(axis=2)


def build_arrays(erp: dict[str, pd.DataFrame]) -> SchoolArrays:
    st = erp["students"].sort_values("student_id").reset_index(drop=True)
    sid = st["student_id"].to_numpy()
    idx = {s: i for i, s in enumerate(sid)}
    n, nD, nW, nS = len(sid), C.N_DAYS, C.N_WEEKS, len(C.SUBJECTS)
    day_of = {C.day_date(d).isoformat(): d for d in range(nD)}

    att = erp["attendance"]
    ai = att["student_id"].map(idx).to_numpy()
    ad = att["date"].map(day_of).to_numpy()
    status = att["status"].to_numpy()
    grid = np.full((n, nD), "", dtype=object)
    grid[ai, ad] = np.where(pd.isna(status), "", status)
    marked = grid != ""
    present = (grid == "present") | (grid == "late")
    absent_any = (grid == "absent") | (grid == "excused")
    unexcused = grid == "absent"
    late_arrival = grid == "late"

    asg = erp["assignments"]
    si = {s: k for k, s in enumerate(C.SUBJECTS)}
    gi = asg["student_id"].map(idx).to_numpy()
    gw = (asg["due_day"].to_numpy() // 5)
    gs = asg["subject"].map(si).to_numpy()
    st_a = asg["status"].to_numpy()
    due = np.zeros((n, nW, nS), np.int16)
    missing = np.zeros_like(due)
    late_sub = np.zeros_like(due)
    np.add.at(due, (gi, gw, gs), 1)
    np.add.at(missing, (gi, gw, gs), (st_a == "missing").astype(np.int16))
    np.add.at(late_sub, (gi, gw, gs), (st_a == "late").astype(np.int16))
    sub_day = asg["submitted_day"].to_numpy()
    ok = sub_day >= 0
    last = np.full((n, nW), -1, np.int32)
    sw = np.minimum(sub_day[ok] // 5, nW - 1)
    np.maximum.at(last, (gi[ok], sw), sub_day[ok])
    last = np.maximum.accumulate(last, axis=1)

    lms = erp["lms"]
    li = lms["student_id"].map(idx).to_numpy()
    ld = lms["date"].map(day_of).to_numpy()
    lms_active = np.zeros((n, nD), bool)
    lms_active[li, ld] = lms["active"].to_numpy() > 0

    ass = erp["assessments"]
    aw = np.zeros((nS, len(C.ASSESSMENT_BASE_WEEKS)), int)
    marks = np.full((n, nS, len(C.ASSESSMENT_BASE_WEEKS)), np.nan)
    for s_name, grp in ass.groupby("subject"):
        s = si[s_name]
        weeks_sorted = sorted(grp["week"].unique())
        aw[s, :] = weeks_sorted
        a_of = {w: k for k, w in enumerate(weeks_sorted)}
        marks[grp["student_id"].map(idx).to_numpy(), s, grp["week"].map(a_of).to_numpy()] = (
            100.0 * grp["marks"].to_numpy() / grp["max_marks"].to_numpy())

    tt = erp["timetable"]
    changes: dict = {}
    for r in tt[tt["effective_from_week"] > 0].itertuples():
        changes.setdefault(r[1], []).append((si[r.subject], int(r.effective_from_week)))

    fees = erp["fees"].sort_values(["student_id", "invoice_id"])
    k = fees.groupby("student_id").size().max()
    fee_due = fees["due_day"].to_numpy().reshape(n, k)
    fee_paid = fees["paid_day"].to_numpy().reshape(n, k)

    adm_year = st["admission_date"].str[:4].astype(int).to_numpy()
    return SchoolArrays(
        sid=sid, names=st["name"].to_numpy(), klass=st["class"].to_numpy(), grade=st["grade"].to_numpy(),
        lang=st["guardian_language"].to_numpy(), new_student=adm_year == C.YEAR_START.year,
        years_enrolled=C.YEAR_START.year - adm_year,
        marked=marked, present=present, absent_any=absent_any, unexcused=unexcused, late_arrival=late_arrival,
        due=due, missing=missing, late_sub=late_sub, last_sub_day=last, lms_active=lms_active,
        marks=marks, assess_week=aw, tt_changes=changes, fee_due=fee_due, fee_paid=fee_paid,
    )


# ---------------------------------------------------------------- windows
class Windows:
    """Cumulative sums over weeks; window sums in O(1) per snapshot."""

    def __init__(self, weekly: np.ndarray):
        self.cs = np.concatenate([np.zeros(weekly.shape[:1] + (1,) + weekly.shape[2:]), np.cumsum(weekly, axis=1)], axis=1)

    def sum(self, t: int, length: int, gap: int = 0) -> np.ndarray:
        hi = t - gap
        if hi < 0:
            return np.zeros_like(self.cs[:, 0])
        lo = max(0, hi - length + 1)
        return self.cs[:, hi + 1] - self.cs[:, lo]


def rate(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)


FEATURES = [
    # attendance (10)
    "att_w4", "att_base8", "att_change", "att_ttd", "att_projection", "weekday_max_share_6w",
    "absences_6w", "unexcused_streak_10d", "late_arrivals_w4", "unexcused_w4",
    # assignments (8)
    "missing_rate_w4", "missing_rate_base8", "missing_change", "late_rate_w4", "late_rate_base8",
    "late_change", "max_subject_missing_w4", "days_since_submission",
    # assessments (9)
    "score_mean", "score_last", "score_min_last3", "score_max_drop", "margin_to_pass",
    "z_in_section", "n_assessments", "core_min_last", "score_trend",
    # engagement (5)
    "lms_w4", "lms_base8", "lms_ratio", "lms_last_week", "days_since_lms",
    # context (9)
    "grade", "weeks_since_admission", "new_student", "week_of_term", "weeks_to_next_exam",
    "timetable_change_3w", "section_slip", "earlier_flags_term", "open_intervention",
]
assert len(FEATURES) == 41

# features that sit under each ranking signal (used to order reasons by attribution)
SIGNAL_FEATURES = {
    "A1": ["att_w4", "att_base8", "att_change"],
    "A2": ["att_ttd", "att_projection"],
    "A3": ["weekday_max_share_6w", "absences_6w"],
    "A4": ["unexcused_streak_10d", "unexcused_w4"],
    "H1": ["missing_rate_w4", "max_subject_missing_w4", "missing_change", "days_since_submission"],
    "H2": ["late_rate_w4", "late_change"],
    "G1": ["score_max_drop", "score_trend", "score_last"],
    "G2": ["score_min_last3", "margin_to_pass", "core_min_last"],
    "E1": ["lms_w4", "lms_ratio", "lms_last_week", "days_since_lms"],
}
