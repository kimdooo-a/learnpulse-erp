"""The 12 signal definitions, each tested against a hand-built case."""
from __future__ import annotations

import numpy as np
import pytest
from learnpulse import config as C
from learnpulse.signals import compute, eligible, severity

from .conftest import absent, make_school


def fired_at(R, sig, t):
    return set(np.where(R.fired[sig][t])[0].tolist())


def test_regular_class_fires_nothing():
    R = compute(make_school())
    for sig in C.RANKING_SIGNALS + C.ANNOTATION_SIGNALS:
        assert R.fired[sig].sum() == 0, sig
    assert R.s1_events == []
    assert R.dq_incomplete == set()


def test_A1_attendance_drop():
    A = make_school()
    for w in range(26, 30):  # 0-based weeks; absent Mon and Tue
        absent(A, 0, w * 5, excused=True)
        absent(A, 0, w * 5 + 1, excused=True)
    R = compute(A)
    assert 0 in fired_at(R, "A1", 29)
    assert R.facts[(29, 0)]["A1"] == {"from": 100, "to": 60, "weeks": 4}
    assert 0 not in fired_at(R, "A1", 25)


def test_A1_needs_twelve_weeks_of_history():
    A = make_school()
    for d in range(30, 40):  # weeks 6-7 (0-based)
        absent(A, 0, d, excused=True)
    R = compute(A)
    assert R.fired["A1"][:11].sum() == 0


def test_A2_minimum_attendance_projection():
    A = make_school()
    for w in range(20, 26):
        for wd in range(3):
            absent(A, 1, w * 5 + wd, excused=True)
    R = compute(A)
    assert 1 in fired_at(R, "A2", 25)
    f = R.facts[(25, 1)]["A2"]
    assert f["mode"] == "projection" and f["projection"] < 75


def test_A3_weekday_cluster():
    A = make_school()
    for w in range(24, 28):
        absent(A, 2, w * 5 + 0, excused=True)  # Mondays
    R = compute(A)
    assert 2 in fired_at(R, "A3", 27)
    assert R.facts[(27, 2)]["A3"] == {"k": 4, "absences": 4, "weekday": "Monday", "weeks": 6}


def test_A4_unexcused_streak_and_excused_does_not_count():
    A = make_school()
    for d in (140, 141, 142):
        absent(A, 3, d)
        absent(A, 4, d, excused=True)
    R = compute(A)
    assert 3 in fired_at(R, "A4", 28)
    assert 4 not in fired_at(R, "A4", 28)
    assert R.facts[(28, 3)]["A4"]["days"] == 3


def test_H1_missing_work_in_one_subject():
    A = make_school()
    A.missing[5, 28, 0] = 2  # two Maths assignments missing
    A.missing[6, 28, 0] = 1  # one is not enough
    R = compute(A)
    assert 5 in fired_at(R, "H1", 28)
    assert 6 not in fired_at(R, "H1", 28)
    assert R.facts[(28, 5)]["H1"]["subject"] == "Maths"


def test_H2_late_work_rise():
    A = make_school()
    A.late_sub[7, 26:30, :] = 1  # 6 of 12 per week late in W4, none before
    R = compute(A)
    assert 7 in fired_at(R, "H2", 29)
    assert R.facts[(29, 7)]["H2"] == {"from": 0, "to": 50}


def test_G1_score_decline_needs_three_earlier_assessments():
    A = make_school()
    A.marks[8, 0, :] = [70, 70, 70, 70, 50, 50]
    R = compute(A)
    t5, t6 = A.assess_week[0, 4], A.assess_week[0, 5]
    assert 8 not in fired_at(R, "G1", t5)  # last two 70/50 = 60 vs 70: -10
    assert 8 in fired_at(R, "G1", t6)  # last two 50/50 vs 70: -20
    assert R.facts[(int(t6), 8)]["G1"] == {"subject": "Maths", "from": 70, "to": 50}


def test_G2_near_pass_mark():
    A = make_school()
    A.marks[9, 1, :3] = [35, 36, 37]
    R = compute(A)
    t3 = A.assess_week[1, 2]
    assert 9 in fired_at(R, "G2", t3)
    assert R.facts[(int(t3), 9)]["G2"]["margin"] == 3


def test_E1_engagement_drop():
    A = make_school()
    A.lms_active[10, 26 * 5:30 * 5] = False
    R = compute(A)
    assert 10 in fired_at(R, "E1", 29)


def test_S1_section_slip_routes_to_head_and_suppresses_G1():
    A = make_school()
    hit = np.arange(0, 28)  # 70% of the class
    A.marks[hit, 0, 4:] -= 25
    R = compute(A)
    assert any(e["subject"] == "Maths" and e["assessment"] == 4 for e in R.s1_events)
    t6 = A.assess_week[0, 5]
    assert R.fired["G1"][t6].sum() == 0  # forty students are not flagged for one hard test


def test_T1_timetable_link_is_annotation_only():
    A = make_school()
    A.tt_changes = {"8-B": [(0, 25)]}
    for w in range(26, 30):
        absent(A, 11, w * 5, excused=True)
        absent(A, 11, w * 5 + 1, excused=True)
    R = compute(A)
    assert 11 in fired_at(R, "T1", 29)
    sev_with = severity(R.fired, 29)[11]
    assert sev_with == pytest.approx(C.WEIGHTS["A1"] + C.WEIGHTS["A2"] * R.fired["A2"][29][11])


def test_F1_fee_delay_is_context_only():
    A = make_school()
    A.fee_paid[12, 0] = -1  # first invoice never paid
    R = compute(A)
    assert R.fired["F1"][12:, 12].all()
    assert not eligible(R.fired, 20)[12]


def test_data_quality_gate_holds_back_a_class():
    A = make_school()
    A.marked[:, 30 * 5:30 * 5 + 2] = False  # 40% of the week unmarked
    R = compute(A)
    assert ("8-B", 30) in R.dq_incomplete


def test_priority_worked_example_from_design():
    # A1 + A3 + H1 -> severity 0.80; model 0.77 -> priority 0.78
    sev = C.WEIGHTS["A1"] + C.WEIGHTS["A3"] + C.WEIGHTS["H1"]
    assert sev == pytest.approx(0.80)
    assert round(C.MIX_MODEL * 0.77 + C.MIX_RULES * sev, 2) == 0.78
