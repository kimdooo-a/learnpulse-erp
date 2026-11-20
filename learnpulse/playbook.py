"""Intervention playbook (R2 section 6). The school edits this table."""
from __future__ import annotations

PLAYBOOK = {
    "attendance": {
        "signals": ["A1", "A4"],
        "action": "5-minute check-in with the student; attendance note to the parent",
        "action_hi": "विद्यार्थी से 5 मिनट की बातचीत करना और आपको उपस्थिति के बारे में सूचित करना",
        "owner": "Class teacher",
        "tracked": "4-week attendance back to 90% or more",
    },
    "weekday": {
        "signals": ["A3"],
        "action": "Ask about that weekday (transport, timetable, duties at home); parent call if it repeats",
        "action_hi": "उस दिन की अनुपस्थिति का कारण (आवागमन, समय-सारिणी, घर की ज़िम्मेदारियाँ) समझना",
        "owner": "Class teacher",
        "tracked": "At most 1 absence on that weekday",
    },
    "minimum_attendance": {
        "signals": ["A2"],
        "action": "Attendance notice with the days still allowed; parent meeting",
        "action_hi": "न्यूनतम उपस्थिति के बारे में आपसे मिलकर बात करना",
        "owner": "Class teacher, academic head",
        "tracked": "Projected term attendance 75% or more",
    },
    "coursework": {
        "signals": ["H1", "H2", "E1"],
        "action": "Catch-up plan with dates; peer tutor or homework slot",
        "action_hi": "तारीख़ों के साथ एक कैच-अप योजना बनाना",
        "owner": "Subject teacher",
        "tracked": "Missing rate below 15%",
    },
    "score_decline": {
        "signals": ["G1"],
        "action": "Re-check the weak topic; remedial slot",
        "action_hi": "कमज़ोर विषय-वस्तु को दोबारा देखना और अतिरिक्त कक्षा देना",
        "owner": "Subject teacher",
        "tracked": "Next assessment within 5 points of the earlier mean",
    },
    "near_pass": {
        "signals": ["G2"],
        "action": "Term remedial plan; parent informed of the plan",
        "action_hi": "इस सत्र के लिए सुधार योजना बनाना",
        "owner": "Subject and class teacher",
        "tracked": "Next two assessments at 40% or more",
    },
    "counsellor": {
        "signals": [],
        "action": "Counsellor referral with the history attached",
        "action_hi": "परामर्शदाता से मिलवाना",
        "owner": "Counsellor",
        "tracked": "Follow-up logged; no ranking signal 4 weeks later",
    },
}

SEVERAL = 4  # this many ranking signals -> counsellor referral
_BY_SIGNAL = {s: key for key, p in PLAYBOOK.items() for s in p["signals"]}


def choose(ordered_signals: list[str]) -> str:
    """Playbook entry for the leading reason (signals arrive ordered by attribution)."""
    ranking = [s for s in ordered_signals if s in _BY_SIGNAL]
    if len(ranking) >= SEVERAL or not ranking:
        return "counsellor"
    return _BY_SIGNAL[ranking[0]]
