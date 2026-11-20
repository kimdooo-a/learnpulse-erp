"""Seeded synthetic school-ERP generator.

Produces the canonical ERP tables for one synthetic school year (no real
person is represented) plus a ground-truth table that is used only for
evaluation. The pipeline never reads the ground truth.

Shape (R2 section 8.1): 1,200 students = grades 6-10 x 6 sections x 40,
40 weeks x 5 days, 6 subjects.

Injected ground truth: about 15% of students follow one of four
trajectories with a known onset week, and two sections suffer a
section-level score slip after a teacher change. Noise (illness waves,
festival weeks, random absences, late marking, missing entries) is added
so that naive thresholds fail.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

TRAJECTORIES = ["disengagement", "weekday", "subject", "acute"]

_FIRST = ["Aarav", "Vivaan", "Aditya", "Arjun", "Sai", "Reyansh", "Ayaan", "Krishna", "Ishaan", "Shaurya", "Atharv", "Kabir", "Dhruv", "Rudra", "Aanya", "Diya", "Saanvi", "Ananya", "Aadhya", "Pari", "Myra", "Ira", "Anika", "Navya", "Kiara", "Riya", "Meera", "Tara", "Zoya", "Ishita", "Rohan", "Kunal", "Nikhil", "Pranav", "Yash", "Neel", "Om", "Dev", "Aryan", "Harsh", "Kavya", "Pooja", "Sneha", "Nisha", "Divya", "Simran", "Imran", "Farhan", "Sara", "Ayesha", "Gurpreet", "Harleen", "Joseph", "Maria", "Arnav", "Tanvi"]
_LAST_INITIAL = list("ABCDGJKMNPRSTV")


def _ramp(t_rel: np.ndarray, weeks: float) -> np.ndarray:
    """0 before onset, rising linearly to 1 over `weeks` weeks."""
    return np.clip((t_rel + 1) / weeks, 0.0, 1.0) * (t_rel >= 0)


def generate(seed: int = C.SEED) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    nW, nD, nS = C.N_WEEKS, C.N_DAYS, len(C.SUBJECTS)

    # ---------------- students ----------------
    rows = []
    for g in C.GRADES:
        for s in C.SECTIONS:
            for k in range(C.STUDENTS_PER_SECTION):
                rows.append((g, s))
    n = len(rows)
    grade = np.array([r[0] for r in rows])
    section = np.array([r[1] for r in rows])
    klass = np.array([f"{g}-{s}" for g, s in rows])
    sid = np.array([f"S{i + 1:04d}" for i in range(n)])
    names = [f"{rng.choice(_FIRST)} {rng.choice(_LAST_INITIAL)}." for _ in range(n)]
    new_student = rng.random(n) < 0.04
    admission = [
        (C.YEAR_START if new else C.YEAR_START.replace(year=C.YEAR_START.year - int(rng.integers(1, 6)))).isoformat()
        for new in new_student
    ]
    lang = np.where(rng.random(n) < 0.45, "hi", "en")
    students = pd.DataFrame({
        "student_id": sid, "name": names, "class": klass, "grade": grade, "section": section,
        "admission_date": admission, "guardian_language": lang,
    })

    # ---------------- latent traits ----------------
    att_p = rng.beta(32, 1.6, n)  # mean presence ~0.95
    miss_p = rng.beta(1.0, 70, n)  # mean missing ~1.4%
    late_p = rng.beta(1.6, 16, n)  # mean late ~9%
    lms_p = rng.beta(5, 4, n)  # active-day probability
    ability = rng.normal(66, 11, n)[:, None] + rng.normal(0, 6, (n, nS))

    # ---------------- ground truth trajectories ----------------
    n_traj = int(round(0.15 * n))
    traj_idx = rng.choice(n, n_traj, replace=False)
    traj_type = np.array([TRAJECTORIES[i % 4] for i in range(n_traj)])
    rng.shuffle(traj_type)
    onset = rng.integers(8, 34, n_traj)  # 0-based onset week
    traj_weekday = rng.choice(5, n_traj, p=[0.5, 0.1, 0.1, 0.1, 0.2])
    traj_subject = rng.integers(0, nS, n_traj)
    acute_len = rng.integers(3, 5, n_traj)

    weeks = np.arange(nW)
    day_week = np.repeat(weeks, 5)
    day_wd = np.tile(np.arange(5), nW)

    # per student x week modifiers
    abs_add = np.zeros((n, nW))
    miss_add = np.zeros((n, nW))
    late_add = np.zeros((n, nW))
    lms_mult = np.ones((n, nW))
    score_shift = np.zeros((n, nW, nS))
    weekday_abs = np.zeros((n, nW, 5))  # absolute override probability on a weekday
    subj_miss_add = np.zeros((n, nW, nS))

    for j, i in enumerate(traj_idx):
        rel = weeks - onset[j]
        kind = traj_type[j]
        if kind == "disengagement":
            r = _ramp(rel, 6)
            abs_add[i] += 0.30 * r
            miss_add[i] += 0.35 * r
            late_add[i] += 0.30 * r
            lms_mult[i] *= 1 - 0.75 * r
            score_shift[i] += (-18 * r)[:, None]
        elif kind == "weekday":
            on = rel >= 0
            weekday_abs[i, on, traj_weekday[j]] = 0.75
            lms_mult[i] *= np.where(on, 0.85, 1.0)
        elif kind == "subject":
            r = _ramp(rel, 3)
            s = traj_subject[j]
            score_shift[i, :, s] += -24 * r
            subj_miss_add[i, :, s] += 0.40 * r
            late_add[i] += 0.10 * r
        elif kind == "acute":
            L = acute_len[j]
            during = (rel >= 0) & (rel < L)
            after = (rel >= L) & (rel < L + 4)
            abs_add[i] += np.where(during, 0.80, 0.0) + np.where(after, 0.08, 0.0)
            miss_add[i] += np.where(during, 0.70, 0.0)
            lms_mult[i] *= np.where(during, 0.10, 1.0)
            score_shift[i] += np.where(during | after, -8.0, 0.0)[:, None]

    # ---------------- section slips after a teacher change ----------------
    classes = sorted(set(klass))
    slip_classes = rng.choice(classes, 2, replace=False)
    slip_subjects = rng.choice([0, 1], 2)  # Maths or Science
    slip_weeks = np.array([17, 23])  # 0-based week of the teacher change
    for c, s, w in zip(slip_classes, slip_subjects, slip_weeks):
        members = np.where(klass == c)[0]
        hit = rng.random(len(members)) < 0.65
        score_shift[members[hit], w:, s] += -15
        score_shift[members[~hit], w:, s] += -4

    # ---------------- attendance ----------------
    p_abs = (1 - att_p)[:, None] + abs_add[:, day_week]
    illness = np.isin(day_week, [12, 13, 27, 28])
    festival = np.isin(day_week, [9, 31])
    p_abs = p_abs + 0.08 * illness[None, :] + 0.05 * festival[None, :]
    # weekday-pattern override
    wd_override = weekday_abs[:, day_week, day_wd]
    p_abs = np.where(wd_override > 0, np.maximum(p_abs, wd_override), p_abs)
    p_abs = np.clip(p_abs, 0, 0.97)
    absent = rng.random((n, nD)) < p_abs
    traj_abs = (abs_add[:, day_week] > 0) | (wd_override > 0)
    excused_p = np.where(traj_abs, 0.08, 0.30)
    excused_p = np.where(illness[None, :], 0.6, excused_p)
    excused = absent & (rng.random((n, nD)) < excused_p)
    late_arr = (~absent) & (rng.random((n, nD)) < 0.03 + 0.5 * late_add[:, day_week] * 0.2)
    status = np.full((n, nD), "present", dtype=object)
    status[late_arr] = "late"
    status[absent] = "absent"
    status[excused] = "excused"
    # 2% random unmarked entries
    unmarked = rng.random((n, nD)) < 0.02
    # late attendance marking: three class-weeks where 30% of entries are still unmarked
    dq_cases = []
    for _ in range(3):
        c = rng.choice(classes)
        w = int(rng.integers(24, 36))
        dq_cases.append((c, w))
        m = (klass == c)[:, None] & (day_week == w)[None, :]
        unmarked |= m & (rng.random((n, nD)) < 0.30)
    status[unmarked] = None
    dates = [C.day_date(d).isoformat() for d in range(nD)]
    attendance = pd.DataFrame({
        "student_id": np.repeat(sid, nD),
        "date": np.tile(dates, n),
        "session": "day",
        "status": status.ravel(),
    })

    # ---------------- LMS ----------------
    p_lms = np.clip(lms_p[:, None] * lms_mult[:, day_week], 0, 1)
    p_lms = np.where(absent, p_lms * 0.3, p_lms)
    active = rng.random((n, nD)) < p_lms
    lms = pd.DataFrame({
        "student_id": np.repeat(sid, nD),
        "date": np.tile(dates, n),
        "active": active.ravel().astype(np.int8),
        "resource_views": (active * rng.poisson(4, (n, nD))).ravel().astype(np.int16),
    })

    # ---------------- assignments ----------------
    K = C.ASSIGNMENTS_PER_SUBJECT_PER_WEEK
    shape = (n, nW, nS, K)
    pm = miss_p[:, None, None, None] + miss_add[:, :, None, None] + subj_miss_add[:, :, :, None]
    pm = np.clip(pm, 0, 0.95)
    missing = rng.random(shape) < pm
    pl = np.clip(late_p[:, None, None, None] + late_add[:, :, None, None], 0, 0.9)
    late = (~missing) & (rng.random(shape) < pl)
    due_wd = np.array([2, 4])[None, None, None, :]
    due_day = weeks[None, :, None, None] * 5 + due_wd
    due_day = np.broadcast_to(due_day, shape)
    delay = np.where(late, rng.integers(1, 6, shape), 0)
    sub_day = np.where(missing, -1, due_day + delay)
    I, Wk, Sb, Kk = np.indices(shape)
    a_status = np.where(missing, "missing", np.where(late, "late", "on_time"))
    subj_names = np.array(C.SUBJECTS)
    assignments = pd.DataFrame({
        "assignment_id": [f"{c}-{s[:3].upper()}-W{w + 1:02d}-{k + 1}" for c, s, w, k in zip(
            klass[I.ravel()], subj_names[Sb.ravel()], Wk.ravel(), Kk.ravel())],
        "student_id": sid[I.ravel()],
        "subject": subj_names[Sb.ravel()],
        "due_day": due_day.ravel().astype(np.int16),
        "submitted_day": sub_day.ravel().astype(np.int16),
        "status": a_status.ravel(),
    })

    # ---------------- assessments ----------------
    a_rows = []
    class_idx = {c: np.where(klass == c)[0] for c in classes}
    for s in range(nS):
        offset = s % 3
        for a, bw in enumerate(C.ASSESSMENT_BASE_WEEKS):
            w = bw + offset
            for c, members in class_idx.items():
                difficulty = rng.normal(0, 1.5)
                marks = ability[members, s] + score_shift[members, w, s] + difficulty + rng.normal(0, 7, len(members))
                marks = np.clip(np.round(marks), 0, 100)
                for i, m in zip(members, marks):
                    a_rows.append((f"{c}-{subj_names[s][:3].upper()}-A{a + 1}", sid[i], subj_names[s], w, float(m), 100.0))
    assessments = pd.DataFrame(a_rows, columns=["assessment_id", "student_id", "subject", "week", "marks", "max_marks"])

    # ---------------- timetable ----------------
    tt_rows = []
    tid = 100
    for c in classes:
        for s in range(nS):
            tid += 1
            tt_rows.append((c, subj_names[s], f"T{tid}", 0))
    for c, s, w in zip(slip_classes, slip_subjects, slip_weeks):
        tid += 1
        tt_rows.append((c, subj_names[s], f"T{tid}", int(w)))
    # unrelated timetable changes (noise)
    for _ in range(4):
        tid += 1
        tt_rows.append((rng.choice(classes), subj_names[int(rng.integers(2, nS))], f"T{tid}", int(rng.integers(8, 32))))
    timetable = pd.DataFrame(tt_rows, columns=["class", "subject", "teacher_id", "effective_from_week"])

    # ---------------- fees ----------------
    f_rows = []
    for i in range(n):
        for k, w in enumerate(C.INVOICE_WEEKS):
            due = w * 5
            r = rng.random()
            if r < 0.05:
                paid = -1
            elif r < 0.13:
                paid = due + int(rng.integers(25, 60))
            else:
                paid = due + int(rng.integers(0, 8))
            f_rows.append((f"INV-{sid[i]}-{k + 1}", sid[i], due, paid))
    fees = pd.DataFrame(f_rows, columns=["invoice_id", "student_id", "due_day", "paid_day"])

    # ---------------- ground truth (evaluation only) ----------------
    truth = pd.DataFrame({
        "student_id": sid[traj_idx],
        "trajectory": traj_type,
        "onset_week": onset + 1,  # 1-based
        "weekday": [C.WEEKDAYS[d] if k == "weekday" else "" for d, k in zip(traj_weekday, traj_type)],
        "subject": [C.SUBJECTS[s] if k == "subject" else "" for s, k in zip(traj_subject, traj_type)],
    })
    section_truth = pd.DataFrame({
        "class": slip_classes, "subject": subj_names[slip_subjects], "change_week": slip_weeks + 1,
    })
    dq_truth = pd.DataFrame(dq_cases, columns=["class", "week0"])
    dq_truth["week"] = dq_truth.pop("week0") + 1

    return {
        "students": students, "attendance": attendance, "assignments": assignments,
        "assessments": assessments, "timetable": timetable, "lms": lms, "fees": fees,
        "truth_trajectories": truth, "truth_sections": section_truth, "truth_data_quality": dq_truth,
    }


ERP_TABLES = ["students", "attendance", "assignments", "assessments", "timetable", "lms", "fees"]
TRUTH_TABLES = ["truth_trajectories", "truth_sections", "truth_data_quality"]
