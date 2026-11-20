from __future__ import annotations

import numpy as np
from learnpulse import config as C
from learnpulse.generate import ERP_TABLES, generate
from learnpulse.pipeline import rank_lists
from learnpulse.playbook import choose
from learnpulse.reasons import merge_name, parent_note, reason_en, render, verify


def test_number_check_accepts_template_and_rejects_invented_number():
    f = {"from": 96, "to": 81, "weeks": 4}
    text = reason_en("A1", f)
    assert text == "Attendance fell from 96% to 81% in 4 weeks."
    assert verify(text, f)
    assert not verify("Attendance fell from 96% to 71% in 4 weeks.", f)


def test_rewriter_output_with_wrong_number_falls_back_to_template():
    f = {"from": 96, "to": 81, "weeks": 4}
    tmpl = reason_en("A1", f)
    good, src = render(tmpl, f, rewriter=lambda s: "In 4 weeks attendance went from 96% to 81%.")
    assert src == "rewritten" and "96%" in good
    bad, src = render(tmpl, f, rewriter=lambda s: "Attendance fell sharply to 18%.")
    assert src == "template" and bad == tmpl
    broken, src = render(tmpl, f, rewriter=lambda s: 1 / 0)
    assert src == "template"


def test_parent_notes_keep_placeholder_until_local_merge():
    f = {"subject": "Maths", "from": 68, "to": 49}
    for lang in ("en", "hi"):
        note = parent_note(lang, "G1", f, "score_decline", "8-B")
        assert "{student}" in note
        assert verify(note, f, extra_text="8-B 5")
        merged = merge_name(note, "Student 7")
        assert "{student}" not in merged and "Student 7" in merged
    assert "गणित" in parent_note("hi", "G1", f, "score_decline", "8-B")


def test_playbook_follows_leading_reason_and_escalates_many_signals():
    assert choose(["A3", "A1"]) == "weekday"
    assert choose(["G1", "G2"]) == "score_decline"
    assert choose(["A1", "A3", "H1", "E1"]) == "counsellor"


def test_top_five_per_class():
    klass = np.array(["A"] * 8 + ["B"] * 3)
    elig = np.array([1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1], bool)
    prio = np.array([.1, .9, .5, .3, .7, .2, .99, .4, .6, .8, .1])
    L = rank_lists(klass, elig, prio)
    assert sorted(L.tolist()) == [1, 2, 3, 4, 7, 8, 10]
    assert 6 not in L and 9 not in L  # not eligible -> never listed, whatever the score


def test_generator_shape_is_reproducible_and_has_no_sensitive_fields():
    a, b = generate(7), generate(7)
    assert len(a["students"]) == 1200
    assert len(a["attendance"]) == 240_000
    assert len(a["assignments"]) == 576_000
    assert len(a["assessments"]) == 43_200
    assert len(a["lms"]) == 240_000
    assert len(a["fees"]) == 4_800
    assert sum(len(a[t]) for t in ERP_TABLES) > 1_100_000
    assert a["attendance"]["status"].equals(b["attendance"]["status"])
    assert len(a["truth_trajectories"]) == 180
    forbidden = {"caste", "religion", "income", "gender", "health", "phone", "address", "dob"}
    for t in ERP_TABLES:
        assert not forbidden & set(a[t].columns)
    assert a["attendance"]["status"].isna().mean() > 0.015  # missing entries are part of the noise
    assert C.day_date(0).weekday() == 0
