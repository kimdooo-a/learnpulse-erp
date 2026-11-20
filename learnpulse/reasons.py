"""Reason sentences and parent-note drafts (R2 section 4.2).

1. Input is a structured fact list with placeholders - never a name or ID.
2. A template renders the default text. An optional `rewriter` (for
   example a tone or translation service) may rewrite it.
3. Code checks that every number in the final text exists in the facts.
   If not, the template text is used.
4. Names are merged locally, after the check.
5. The teacher reviews and approves; nothing is sent automatically.
"""
from __future__ import annotations

import re
from collections.abc import Callable

from .playbook import PLAYBOOK

SUBJECT_HI = {
    "Maths": "गणित", "Science": "विज्ञान", "English": "अंग्रेज़ी", "Hindi": "हिंदी",
    "Social Science": "सामाजिक विज्ञान", "Computer": "कंप्यूटर",
}
WEEKDAY_HI = {"Monday": "सोमवार", "Tuesday": "मंगलवार", "Wednesday": "बुधवार", "Thursday": "गुरुवार", "Friday": "शुक्रवार"}

SIGNAL_TITLES = {
    "A1": "Attendance drop", "A2": "Minimum-attendance risk", "A3": "Weekday cluster", "A4": "Absence streak",
    "H1": "Missing work", "H2": "Late-work rise", "G1": "Score decline", "G2": "Near pass mark",
    "E1": "Engagement drop", "S1": "Section slip", "T1": "Timetable link", "F1": "Fee-delay context",
}


def reason_en(signal: str, f: dict) -> str:
    if signal == "A1":
        return f"Attendance fell from {f['from']}% to {f['to']}% in {f['weeks']} weeks."
    if signal == "A2":
        if f["mode"] == "projection":
            return f"At this pace attendance ends the term at {f['projection']}%, below the {f['minimum']}% minimum."
        return f"Term-to-date attendance is {f['ttd']}%, below {f['threshold']}%."
    if signal == "A3":
        return f"{f['k']} of {f['absences']} absences in {f['weeks']} weeks were on {f['weekday']}s."
    if signal == "A4":
        return f"Absent {f['days']} school days in a row, no reason recorded."
    if signal == "H1":
        if f["mode"] == "subject":
            return f"{f['missing']} {f['subject']} assignments missing in the last {f['weeks']} weeks."
        return f"{f['missing']} of {f['due']} assignments ({f['rate']}%) missing in the last {f['weeks']} weeks."
    if signal == "H2":
        return f"Late submissions rose from {f['from']}% to {f['to']}%."
    if signal == "G1":
        return f"{f['subject']} average dropped from {f['from']}% to {f['to']}%."
    if signal == "G2":
        side = "above" if f["margin"] >= 0 else "below"
        return f"{f['subject']} average is {f['avg']}%, {abs(f['margin'])} points {side} the {f['pass_mark']}% pass mark."
    if signal == "E1":
        return f"LMS activity fell from {f['from']} to {f['to']} active days."
    if signal == "S1":
        return f"{f['class']} {f['subject']} median fell from {f['from']}% to {f['to']}%; {f['dropped']} of {f['size']} students dropped."
    if signal == "T1":
        return f"The drop began within {f['weeks']} weeks of the timetable change in week {f['change_week']}."
    if signal == "F1":
        return f"Fee invoice overdue {f['days']} days."
    raise KeyError(signal)


def fact_hi(signal: str, f: dict) -> str:
    if signal == "A1":
        return f"पिछले {f['weeks']} सप्ताह में {{student}} की उपस्थिति {f['from']}% से घटकर {f['to']}% हो गई है"
    if signal == "A2":
        if f["mode"] == "projection":
            return f"इसी गति से सत्र के अंत तक {{student}} की उपस्थिति {f['projection']}% रहेगी, जो न्यूनतम {f['minimum']}% से कम है"
        return f"इस सत्र में अब तक {{student}} की उपस्थिति {f['ttd']}% है"
    if signal == "A3":
        return f"पिछले {f['weeks']} सप्ताह की {f['absences']} अनुपस्थितियों में से {f['k']} {WEEKDAY_HI[f['weekday']]} को थीं"
    if signal == "A4":
        return f"{{student}} लगातार {f['days']} स्कूली दिनों से अनुपस्थित है और कोई कारण दर्ज नहीं है"
    if signal == "H1":
        if f["mode"] == "subject":
            return f"पिछले {f['weeks']} सप्ताह में {SUBJECT_HI[f['subject']]} के {f['missing']} गृहकार्य जमा नहीं हुए"
        return f"पिछले {f['weeks']} सप्ताह में {f['due']} में से {f['missing']} गृहकार्य जमा नहीं हुए"
    if signal == "H2":
        return f"देर से जमा होने वाले गृहकार्य {f['from']}% से बढ़कर {f['to']}% हो गए हैं"
    if signal == "G1":
        return f"{SUBJECT_HI[f['subject']]} में औसत अंक {f['from']}% से घटकर {f['to']}% हो गए हैं"
    if signal == "G2":
        return f"{SUBJECT_HI[f['subject']]} में पिछले तीन मूल्यांकनों का औसत {f['avg']}% है"
    if signal == "E1":
        return f"ऑनलाइन अध्ययन (LMS) में सक्रिय दिन {f['from']} से घटकर {f['to']} रह गए हैं"
    raise KeyError(signal)


def parent_note(lang: str, signal: str, f: dict, playbook_key: str, klass: str) -> str:
    """Template parent note with the {student} placeholder still in place."""
    p = PLAYBOOK[playbook_key]
    if lang == "hi":
        return (
            f"{{student}} के अभिभावक, नमस्ते।\n\n"
            f"हमने देखा है कि {fact_hi(signal, f)}। "
            f"हम {p['action_hi']} चाहते हैं। "
            f"कृपया अपनी सुविधा अनुसार कक्षा अध्यापक से संपर्क करें।\n\n"
            f"कक्षा अध्यापक, कक्षा {klass}"
        )
    fact = reason_en(signal, f)
    return (
        f"Dear Parent/Guardian of {{student}},\n\n"
        f"We would like to share something we noticed about {{student}} at school. {fact} "
        f"Our suggested first step: {p['action'].lower()}. "
        f"Please contact the class teacher at a convenient time.\n\n"
        f"Class Teacher, Class {klass}"
    )


_NUM = re.compile(r"\d+(?:\.\d+)?")


def numbers(text: str) -> list[str]:
    return _NUM.findall(text)


def allowed_numbers(facts: dict, extra_text: str = "") -> set[str]:
    out: set[str] = set()
    for v in facts.values():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            v = abs(v)
            out.add(str(int(v)) if float(v).is_integer() else str(v))
        elif isinstance(v, str):
            out.update(numbers(v))
    out.update(numbers(extra_text))
    return out


def verify(text: str, facts: dict, extra_text: str = "") -> bool:
    """True when every number in `text` exists in the fact list."""
    allowed = allowed_numbers(facts, extra_text)
    return all(x in allowed for x in numbers(text))


def render(template_text: str, facts: dict, *, extra_text: str = "",
           rewriter: Callable[[str], str] | None = None) -> tuple[str, str]:
    """Return (text, source) where source is 'template' or 'rewritten'."""
    if rewriter is not None:
        try:
            candidate = rewriter(template_text)
        except Exception:
            candidate = None
        if candidate and verify(candidate, facts, extra_text):
            return candidate, "rewritten"
    return template_text, "template"


def merge_name(text: str, name: str) -> str:
    """Step 4: names are merged locally, after generation and checking."""
    return text.replace("{student}", name)
