from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from learnpulse import config as C
from learnpulse.features import SchoolArrays


def make_school(n: int = 40, klass: str = "8-B") -> SchoolArrays:
    """A perfectly regular class: everyone present, all work on time, flat scores."""
    D, W, S = C.N_DAYS, C.N_WEEKS, len(C.SUBJECTS)
    rng = np.random.default_rng(0)
    aw = np.array([[w + s % 3 for w in C.ASSESSMENT_BASE_WEEKS] for s in range(S)])
    lms = np.zeros((n, D), bool)
    lms[:, ::2] = True  # active every other school day
    fee_due = np.array([[w * 5 for w in C.INVOICE_WEEKS]] * n)
    return SchoolArrays(
        sid=np.array([f"S{i + 1:04d}" for i in range(n)]), names=np.array([f"Student {i + 1}" for i in range(n)]),
        klass=np.array([klass] * n), grade=np.full(n, 8), lang=np.array(["en"] * n),
        new_student=np.zeros(n, bool), years_enrolled=np.ones(n, int),
        marked=np.ones((n, D), bool), present=np.ones((n, D), bool), absent_any=np.zeros((n, D), bool),
        unexcused=np.zeros((n, D), bool), late_arrival=np.zeros((n, D), bool),
        due=np.full((n, W, S), 2, np.int16), missing=np.zeros((n, W, S), np.int16), late_sub=np.zeros((n, W, S), np.int16),
        last_sub_day=np.tile(np.arange(W) * 5 + 4, (n, 1)), lms_active=lms,
        marks=70 + rng.integers(-2, 3, (n, S, len(C.ASSESSMENT_BASE_WEEKS))).astype(float), assess_week=aw,
        tt_changes={}, fee_due=fee_due, fee_paid=fee_due.copy(),
    )


def absent(A: SchoolArrays, i: int, day: int, excused: bool = False) -> None:
    A.present[i, day] = False
    A.absent_any[i, day] = True
    A.unexcused[i, day] = not excused


@pytest.fixture(scope="session")
def built_db(tmp_path_factory) -> Path:
    from learnpulse.pipeline import build
    root = tmp_path_factory.mktemp("lp")
    db = root / "learnpulse.db"
    build(db, report_dir=root / "reports")
    return db
