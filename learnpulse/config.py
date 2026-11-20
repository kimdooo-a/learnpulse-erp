"""School calendar, thresholds and weights.

Every threshold lives here so a school can change it without touching code.
Week numbers in the UI and reports are 1-based (week 1 .. week 40);
arrays inside the pipeline are 0-based (t = 0 .. 39).
"""
from __future__ import annotations

import datetime as dt

# --- calendar -------------------------------------------------------------
N_WEEKS = 40
DAYS_PER_WEEK = 5
N_DAYS = N_WEEKS * DAYS_PER_WEEK
YEAR_START = dt.date(2025, 4, 7)  # a Monday; synthetic academic year
TERMS = [(0, 20), (20, 40)]  # [start, end) in 0-based weeks
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

# --- school shape ---------------------------------------------------------
GRADES = [6, 7, 8, 9, 10]
SECTIONS = ["A", "B", "C", "D", "E", "F"]
STUDENTS_PER_SECTION = 40
SUBJECTS = ["Maths", "Science", "English", "Hindi", "Social Science", "Computer"]
CORE_SUBJECTS = ["Maths", "Science", "English"]
ASSIGNMENTS_PER_SUBJECT_PER_WEEK = 2
ASSESSMENT_BASE_WEEKS = [5, 11, 17, 23, 29, 35]  # 0-based; plus a per-subject offset
INVOICE_WEEKS = [2, 12, 22, 32]

# --- windows --------------------------------------------------------------
W4 = 4  # last 4 school weeks
BASE8 = 8  # the 8 weeks before W4
MIN_WEEKS_FOR_SIGNALS = 4
MIN_WEEKS_FOR_BASELINE = 12

# --- thresholds (R2 section 4) -------------------------------------------
PASS_MARK = 33.0
MIN_ATTENDANCE = 0.75
T = {
    "A1_drop_pts": 10.0,
    "A1_max_w4": 0.90,
    "A2_ttd": 0.85,
    "A2_projection": MIN_ATTENDANCE,
    "A3_min_absences": 3,
    "A3_share": 0.60,
    "A3_weeks": 6,
    "A4_streak": 3,
    "A4_lookback_days": 10,
    "H1_subject_missing": 2,
    "H1_overall_rate": 0.30,
    "H1_min_due": 5,
    "H2_rate": 0.40,
    "H2_rise_pts": 20.0,
    "G1_drop_pts": 15.0,
    "G1_min_earlier": 3,
    "G2_margin_pts": 7.0,
    "E1_ratio": 0.50,
    "E1_min_base_days_per_4w": 4,
    "S1_median_drop": 10.0,
    "S1_share": 0.40,
    "S1_alert_weeks": 3,
    "T1_weeks": 3,
    "F1_days": 30,
    "DQ_min_marked": 0.90,
}

RANKING_SIGNALS = ["A1", "A2", "A3", "A4", "H1", "H2", "G1", "G2", "E1"]
ANNOTATION_SIGNALS = ["T1", "F1"]
SECTION_SIGNALS = ["S1"]
ALL_SIGNALS = RANKING_SIGNALS + SECTION_SIGNALS + ANNOTATION_SIGNALS

WEIGHTS = {
    "A1": 0.35, "A2": 0.40, "A3": 0.20, "A4": 0.30, "H1": 0.25,
    "H2": 0.15, "G1": 0.30, "G2": 0.35, "E1": 0.15,
}
MIX_MODEL = 0.6
MIX_RULES = 0.4

# --- list discipline ------------------------------------------------------
LIST_SIZE = 5
SNOOZE_WEEKS = 2
FOLLOWUP_WEEKS = 4

# --- evaluation split (1-based weeks, R2 section 8.2) -----------------------
TRAIN_WEEKS = (5, 22)
CALIB_WEEKS = (19, 22)  # tail of the training window, used for isotonic calibration
TEST_WEEKS = (29, 34)
LABEL_HORIZON = 6
LABEL_ATTENDANCE = 0.80
LABEL_MISSING = 0.50

SEED = 2026


def day_date(day: int) -> dt.date:
    """Calendar date of a 0-based school day (Mon-Fri)."""
    week, wd = divmod(day, DAYS_PER_WEEK)
    return YEAR_START + dt.timedelta(days=7 * week + wd)


def term_of(t: int) -> tuple[int, int]:
    for a, b in TERMS:
        if a <= t < b:
            return a, b
    raise ValueError(t)
