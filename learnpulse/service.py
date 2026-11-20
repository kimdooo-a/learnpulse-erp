"""Service layer behind the API: weekly lists, decisions, outcomes."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import store
from .features import FEATURES, build_arrays
from .pipeline import rank_lists
from .playbook import PLAYBOOK, choose
from .reasons import SIGNAL_TITLES, merge_name, parent_note, reason_en, render
from .signals import compute

DISMISS_REASONS = ["Known illness", "Family event", "Already being supported", "Data error", "Other"]


@dataclass
class User:
    role: str  # teacher | head | counsellor
    klass: str | None = None

    @classmethod
    def parse(cls, header: str | None) -> User:
        """Demo identity from the X-Demo-User header: 'teacher:8-B', 'head' or 'counsellor'."""
        if not header:
            raise PermissionError("missing X-Demo-User header")
        role, _, klass = header.partition(":")
        if role not in ("teacher", "head", "counsellor"):
            raise PermissionError(f"unknown role {role!r}")
        if role == "teacher" and not klass:
            raise PermissionError("teacher needs a class, e.g. teacher:8-B")
        return cls(role=role, klass=klass or None)

    def can_see_class(self, klass: str) -> bool:
        return self.role == "head" or (self.role == "teacher" and self.klass == klass)


class LearnPulseService:
    def __init__(self, db: Path, rewriter=None):
        self.db = Path(db)
        self.rewriter = rewriter
        self.A = build_arrays(store.read_erp(self.db))
        self.R = compute(self.A)
        self.idx = {s: i for i, s in enumerate(self.A.sid)}
        sc = store.read_table(self.db, "weekly_scores").sort_values(["week", "student_id"])
        n = self.A.n
        self.prob = sc["model_prob"].to_numpy().reshape(C.N_WEEKS, n)
        self.sev = sc["severity"].to_numpy().reshape(C.N_WEEKS, n)
        self.priority = sc["priority"].to_numpy().reshape(C.N_WEEKS, n)
        self.eligible = sc["eligible"].to_numpy().reshape(C.N_WEEKS, n).astype(bool)
        ro = store.read_table(self.db, "reason_order").sort_values(["week", "student_id", "rank"])
        self.order = {k: list(g["signal"]) for k, g in ro.groupby(["week", "student_id"])}
        self.s1 = store.read_table(self.db, "section_alerts")
        self.classes = sorted(set(self.A.klass))
        self.con = store.connect(self.db)

    # ------------------------------------------------------------ helpers
    def _facts(self, week: int, i: int) -> dict:
        return self.R.facts.get((week - 1, i), {})

    def _decisions(self, klass: str | None = None) -> pd.DataFrame:
        q = "SELECT * FROM decisions" + (" WHERE class = ?" if klass else "") + " ORDER BY id"
        df = pd.read_sql_query(q, self.con, params=(klass,) if klass else ())
        return df.astype(object).where(pd.notna(df), None)

    def _audit(self, user: User, action: str, detail: dict) -> None:
        self.con.execute("INSERT INTO audit_log(role, action, detail) VALUES (?,?,?)",
                         (f"{user.role}:{user.klass or ''}", action, json.dumps(detail)))
        self.con.commit()

    def meta(self) -> dict:
        return {
            "classes": self.classes, "first_week": C.MIN_WEEKS_FOR_SIGNALS + 1, "last_week": C.N_WEEKS,
            "demo_week": 30, "list_size": C.LIST_SIZE, "signals": SIGNAL_TITLES,
            "dismiss_reasons": DISMISS_REASONS, "synthetic": True,
        }

    # ------------------------------------------------------------ weekly list
    def _excluded(self, klass: str, week: int) -> tuple[np.ndarray, dict, dict]:
        """Students hidden this week (snoozed or in monitoring) and decisions made this week."""
        n = self.A.n
        excl = np.zeros(n, bool)
        this_week, escalate = {}, {}
        for d in self._decisions(klass).itertuples():
            i = self.idx[d.student_id]
            if d.week == week:
                this_week[d.student_id] = d
                continue
            if d.week > week:
                continue
            if d.action == "dismiss" and week - d.week <= C.SNOOZE_WEEKS:
                excl[i] = True
            if d.action in ("accept", "edit"):
                if week - d.week <= C.FOLLOWUP_WEEKS:
                    excl[i] = True
                else:
                    out = self._outcome_status(d)
                    if out["status"] == "not recovered":
                        escalate[d.student_id] = d.week
        return excl, this_week, escalate

    def weekly_list(self, user: User, klass: str, week: int) -> dict:
        if not user.can_see_class(klass):
            raise PermissionError("not allowed to see this class")
        t = week - 1
        members = self.A.klass == klass
        marked = self.A.weekly(self.A.marked)[members, t].sum() / (members.sum() * C.DAYS_PER_WEEK)
        base = {"class": klass, "week": week, "attendance_marked": round(float(marked), 3)}
        if (klass, t) in self.R.dq_incomplete:
            return base | {"status": "data_incomplete", "students": [],
                           "message": f"Only {marked:.0%} of this week's attendance is marked. Flags are held back until the register is complete."}
        excl, this_week, escalate = self._excluded(klass, week)
        listed = rank_lists(self.A.klass, self.eligible[t] & members, self.priority[t], excluded=excl)
        listed = sorted(listed, key=lambda i: -self.priority[t, i])
        items = []
        for rank, i in enumerate(listed, start=1):
            items.append(self._card(user, week, i, rank, this_week.get(self.A.sid[i]), escalate.get(self.A.sid[i])))
        flagged = int((self.eligible[t] & members).sum())
        return base | {"status": "ok", "flagged_in_class": flagged, "students": items}

    def _card(self, user: User, week: int, i: int, rank: int, decision=None, escalated_from=None) -> dict:
        sid = self.A.sid[i]
        facts = self._facts(week, i)
        ordered = self.order.get((week, sid), [s for s in facts if s in C.RANKING_SIGNALS])
        pkey = "counsellor" if escalated_from else choose(ordered)
        reasons = [{"signal": s, "title": SIGNAL_TITLES[s], "text": reason_en(s, facts[s])} for s in ordered[:3]]
        notes = []
        if "T1" in facts:
            notes.append({"signal": "T1", "title": SIGNAL_TITLES["T1"], "text": reason_en("T1", facts["T1"])})
        if "F1" in facts and user.role == "head":
            notes.append({"signal": "F1", "title": SIGNAL_TITLES["F1"], "text": reason_en("F1", facts["F1"]), "restricted": True})
        return {
            "rank": rank, "student_id": sid, "name": self.A.names[i], "class": self.A.klass[i],
            "priority": round(float(self.priority[week - 1, i]), 2),
            "model_probability": round(float(self.prob[week - 1, i]), 2),
            "rule_severity": round(float(self.sev[week - 1, i]), 2),
            "signals": ordered, "reasons": reasons, "annotations": notes,
            "playbook": {"key": pkey, **{k: PLAYBOOK[pkey][k] for k in ("action", "owner", "tracked")}},
            "escalated_from_week": escalated_from,
            "decision": None if decision is None else {"action": decision.action, "step": decision.step,
                                                         "dismiss_reason": decision.dismiss_reason},
        }

    # ------------------------------------------------------------ detail
    def student(self, user: User, sid: str, week: int) -> dict:
        i = self.idx.get(sid)
        if i is None:
            raise KeyError(sid)
        klass = self.A.klass[i]
        referred = not self._decisions(klass).query("student_id == @sid and playbook_key == 'counsellor'").empty
        if not (user.can_see_class(klass) or (user.role == "counsellor" and referred)):
            raise PermissionError("not allowed to see this student")
        t = week - 1
        facts = self._facts(week, i)
        ordered = self.order.get((week, sid), [s for s in facts if s in C.RANKING_SIGNALS])
        pkey = choose(ordered)
        wm, wp = self.A.weekly(self.A.marked)[i], self.A.weekly(self.A.present)[i]
        trend = [None if wm[w] == 0 else round(float(wp[w] / wm[w]), 3) for w in range(t + 1)]
        due, miss = self.A.due[i].sum(axis=1), self.A.missing[i].sum(axis=1)
        lms = self.A.weekly(self.A.lms_active)[i]
        scores = {}
        for s, subj in enumerate(C.SUBJECTS):
            scores[subj] = [{"week": int(self.A.assess_week[s, a]) + 1, "score": float(self.A.marks[i, s, a])}
                            for a in range(self.A.marks.shape[2]) if self.A.assess_week[s, a] <= t]
        drafts = {}
        lead = ordered[0] if ordered else None
        if lead:
            for lang in ("en", "hi"):
                tmpl = parent_note(lang, lead, facts[lead], pkey, klass)
                text, source = render(tmpl, facts[lead], extra_text=PLAYBOOK[pkey]["action"] + PLAYBOOK[pkey]["action_hi"] + klass,
                                      rewriter=self.rewriter)
                drafts[lang] = {"text": merge_name(text, self.A.names[i]), "source": source}
        history = self._decisions(klass).query("student_id == @sid").to_dict("records")
        return {
            "student_id": sid, "name": self.A.names[i], "class": klass, "week": week,
            "guardian_language": self.A.lang[i],
            "priority": round(float(self.priority[t, i]), 2), "model_probability": round(float(self.prob[t, i]), 2),
            "rule_severity": round(float(self.sev[t, i]), 2),
            "facts": [{"signal": s, "title": SIGNAL_TITLES[s], "text": reason_en(s, facts[s]), "values": facts[s]}
                      for s in ordered] + ([{"signal": "T1", "title": SIGNAL_TITLES["T1"], "text": reason_en("T1", facts["T1"]), "values": facts["T1"]}] if "T1" in facts else []),
            "attendance_trend": trend,
            "missing_trend": [None if due[w] == 0 else round(float(miss[w] / due[w]), 3) for w in range(t + 1)],
            "lms_trend": [int(x) for x in lms[: t + 1]],
            "scores": scores,
            "playbook": {"key": pkey, **{k: PLAYBOOK[pkey][k] for k in ("action", "owner", "tracked")}},
            "parent_drafts": drafts,
            "history": history,
        }

    # ------------------------------------------------------------ decisions
    def decide(self, user: User, body: dict) -> dict:
        sid, week, action = body.get("student_id"), int(body.get("week", 0)), body.get("action")
        if sid not in self.idx:
            raise KeyError(sid)
        klass = self.A.klass[self.idx[sid]]
        if not (user.role == "teacher" and user.klass == klass) and user.role != "head":
            raise PermissionError("only the class teacher or the academic head can decide")
        if action not in ("accept", "edit", "dismiss"):
            raise ValueError("action must be accept, edit or dismiss")
        if action == "dismiss" and not body.get("dismiss_reason"):
            raise ValueError("dismiss needs a reason")
        if action == "edit" and not (body.get("step") or body.get("note")):
            raise ValueError("edit needs a changed step or note")
        facts = self._facts(week, self.idx[sid])
        ordered = self.order.get((week, sid), [s for s in facts if s in C.RANKING_SIGNALS])
        pkey = body.get("playbook_key") or choose(ordered)
        if pkey not in PLAYBOOK:
            raise ValueError("unknown playbook key")
        step = body.get("step") or (PLAYBOOK[pkey]["action"] if action != "dismiss" else None)
        cur = self.con.execute(
            "INSERT INTO decisions(student_id, class, week, action, playbook_key, step, note, dismiss_reason, role) VALUES (?,?,?,?,?,?,?,?,?)",
            (sid, klass, week, action, pkey, step, body.get("note"), body.get("dismiss_reason"), user.role))
        self.con.commit()
        self._audit(user, "decision", {"student_id": sid, "week": week, "action": action})
        return {"id": cur.lastrowid, "student_id": sid, "week": week, "action": action, "playbook_key": pkey, "step": step}

    def reset(self, user: User) -> None:
        if user.role != "head":
            raise PermissionError("head only")
        self.con.execute("DELETE FROM decisions")
        self.con.commit()
        self._audit(user, "reset", {})

    # ------------------------------------------------------------ outcomes
    def _outcome_status(self, d) -> dict:
        i = self.idx[d.student_id]
        t = d.week - 1
        end = t + C.FOLLOWUP_WEEKS
        key = d.playbook_key
        facts = self._facts(d.week, i)
        fi = {f: k for k, f in enumerate(FEATURES)}
        out = {"follow_up_weeks": [d.week + 1, d.week + C.FOLLOWUP_WEEKS]}
        if end >= C.N_WEEKS:
            return out | {"status": "pending", "measure": None}
        w = slice(t + 1, end + 1)
        if key == "attendance":
            v = float(self.R.feats[end, i, fi["att_w4"]])
            ok, measure = v >= 0.90, f"4-week attendance {v:.0%}"
        elif key == "weekday":
            wd = C.WEEKDAYS.index(facts.get("A3", {}).get("weekday", "Monday"))
            k = int(self.A.absent_any.reshape(self.A.n, C.N_WEEKS, 5)[i, w, wd].sum())
            ok, measure = k <= 1, f"{k} absences on {C.WEEKDAYS[wd]}s"
        elif key == "minimum_attendance":
            v = float(self.R.feats[end, i, fi["att_projection"]])
            ok, measure = v >= C.MIN_ATTENDANCE, f"projected term attendance {v:.0%}"
        elif key == "coursework":
            due, miss = self.A.due[i, w].sum(), self.A.missing[i, w].sum()
            v = miss / max(due, 1)
            ok, measure = v < 0.15, f"missing rate {v:.0%}"
        elif key in ("score_decline", "near_pass"):
            sig = "G1" if key == "score_decline" else "G2"
            subj = facts.get(sig, {}).get("subject", "Maths")
            s = C.SUBJECTS.index(subj)
            nxt = [a for a in range(self.A.marks.shape[2]) if self.A.assess_week[s, a] > t]
            need = 1 if key == "score_decline" else 2
            if len(nxt) < need:
                return out | {"status": "pending", "measure": f"waiting for the next {subj} assessment"}
            vals = [float(self.A.marks[i, s, a]) for a in nxt[:need]]
            if key == "score_decline":
                ok = vals[0] >= facts["G1"]["from"] - 5
            else:
                ok = all(v >= 40 for v in vals)
            measure = f"next {subj} " + ", ".join(f"{v:.0f}%" for v in vals)
            out["follow_up_weeks"] = [d.week + 1, int(self.A.assess_week[s, nxt[need - 1]]) + 1]
        else:  # counsellor
            fired = [s for s in C.RANKING_SIGNALS if self.R.fired[s][end, i]]
            ok, measure = not fired, "no ranking signal" if not fired else "still: " + ", ".join(fired)
        return out | {"status": "recovered" if ok else "not recovered", "measure": measure}

    def outcomes(self, user: User, klass: str | None) -> dict:
        if user.role == "teacher":
            klass = user.klass
        elif user.role == "counsellor":
            klass = None
        dec = self._decisions(klass).rename(columns={"class": "klass"})
        rows = []
        for d in dec.itertuples():
            if d.action == "dismiss":
                rows.append({"student_id": d.student_id, "name": self.A.names[self.idx[d.student_id]], "class": d.klass,
                             "week": d.week, "action": d.action, "playbook_key": None,
                             "dismiss_reason": d.dismiss_reason, "status": "dismissed", "measure": None})
                continue
            if user.role == "counsellor" and d.playbook_key != "counsellor":
                continue
            o = self._outcome_status(d)
            rows.append({"student_id": d.student_id, "name": self.A.names[self.idx[d.student_id]], "class": d.klass,
                         "week": d.week, "action": d.action, "playbook_key": d.playbook_key,
                         "step": d.step, **o})
        summary = {}
        for r in rows:
            if r["playbook_key"]:
                s = summary.setdefault(r["playbook_key"], {"recovered": 0, "not recovered": 0, "pending": 0})
                s[r["status"]] += 1
        shown = len(rows)
        acted = sum(1 for r in rows if r["action"] in ("accept", "edit"))
        return {"class": klass, "decisions": rows, "by_playbook_action": summary,
                "acceptance_rate": round(acted / shown, 2) if shown else None,
                "note": "Follow-up indicators are computed from the synthetic school year (the weeks after the decision)."}

    def section_alerts(self, user: User, week: int) -> list[dict]:
        if user.role != "head":
            raise PermissionError("section alerts go to the academic head")
        out = []
        for r in self.s1.to_dict("records"):
            if r["week"] > week:
                continue
            f = {"class": r["class"], "subject": r["subject"], "from": int(r["from"]), "to": int(r["to"]),
                 "dropped": int(r["dropped"]), "size": int(r["size"])}
            out.append({"week": int(r["week"]), **f, "text": reason_en("S1", f),
                        "active": week - r["week"] < C.T["S1_alert_weeks"],
                        "action": "Review teaching plan and test difficulty; re-teach the topic. "
                                  "Student-level score-decline flags are suppressed for this subject."})
        return sorted(out, key=lambda x: -x["week"])
