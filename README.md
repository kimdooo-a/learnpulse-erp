# LearnPulse

**An early-warning and intervention copilot for school ERPs — working prototype.**

Once a week LearnPulse reads a school ERP's own records (attendance, assignments, assessments, LMS activity) and gives each class teacher a list of at most five students: *who needs attention, why (2–3 concrete facts), and what to do first*. The teacher accepts, edits or dismisses each suggestion; every decision is logged and the flagged indicators are followed for four weeks.

> **All data in this repository is synthetic.** The generator creates a school of 1,200 fictional students. No real student, school or person is represented, and the numbers in `reports/` describe the synthetic dataset, not a real-school result.

Demo video: `DEMO_VIDEO_URL`

## Quick start

Requires Python 3.11 or newer.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
python -m learnpulse up          # builds the synthetic school on first run (~10 s), then serves
```

Open <http://127.0.0.1:8000/>. Choose a role (class teacher, academic head, counsellor), a class and a week.

Other commands:

```bash
python -m learnpulse build       # regenerate data, rerun the weekly pipeline, rewrite reports/evaluation.*
python -m learnpulse serve       # serve an existing build
pytest -q                        # 28 tests: the 12 signal definitions, text guardrails, data, API roles
docker compose up --build        # same app in a container
```

## What runs

```
synthetic ERP generator ──> canonical ERP tables (SQLite) ──read-only connector──> weekly snapshot features (41)
      ──> 12 rule detectors ──> ranking model (LightGBM + isotonic calibration) ──> priority = 0.6·model + 0.4·rules
      ──> top 5 per class, reasons ordered by feature attribution ──> FastAPI ──> teacher panel
teacher decision ──> decision log ──> 4-week outcome tracker ──> outcome view
```

| Module | File |
|---|---|
| Synthetic school generator (seeded) | `learnpulse/generate.py` |
| Storage and read-only ERP connector | `learnpulse/store.py` |
| Dense arrays and 41 weekly features | `learnpulse/features.py` |
| 12 signals, data-quality gate, labels | `learnpulse/signals.py` |
| Ranking model | `learnpulse/model.py` |
| Weekly batch, ranking, evaluation | `learnpulse/pipeline.py` |
| Reason text, parent notes (English, Hindi), number check | `learnpulse/reasons.py` |
| Intervention playbook | `learnpulse/playbook.py` |
| Lists, decisions, snooze, monitoring, outcomes, roles | `learnpulse/service.py` |
| HTTP API (OpenAPI at `/docs`) | `learnpulse/api.py` |
| Teacher panel | `learnpulse/static/` |

## The 12 signals

| ID | Signal | Fires when |
|---|---|---|
| A1 | Attendance drop | last-4-week rate fell ≥ 10 pts vs the 8 weeks before, and is below 90% |
| A2 | Minimum-attendance risk | term-to-date below 85%, or projected below 75% at the current pace |
| A3 | Weekday cluster | ≥ 3 absences in 6 weeks and ≥ 60% of them on one weekday |
| A4 | Absence streak | ≥ 3 consecutive unexcused school days (last 10 days) |
| H1 | Missing work | ≥ 2 missing in one subject, or ≥ 30% overall (≥ 5 due), last 4 weeks |
| H2 | Late-work rise | late rate ≥ 40% and up ≥ 20 pts |
| G1 | Score decline | mean of last 2 assessments ≥ 15 pts below the student's earlier mean (≥ 3 earlier) |
| G2 | Near pass mark | mean of last 3 assessments in a subject below pass mark + 7 |
| E1 | Engagement drop | LMS active days ≤ 50% of the student's baseline |
| S1 | Section slip | a section's median fell ≥ 10 pts and ≥ 40% of students dropped ≥ 10 — routed to the academic head, suppresses G1 |
| T1 | Timetable link | A1/G1 began within 3 weeks of a timetable change — annotation only |
| F1 | Fee-delay context | invoice overdue > 30 days — annotation only, academic head only, never used for ranking |

All thresholds and weights are in `learnpulse/config.py`. Each signal has a hand-built unit test in `tests/test_signals.py`.

## Results on the synthetic dataset

From `reports/evaluation.md` (seed 2026; regenerate with `python -m learnpulse build`). Time-ordered split: fit on weeks 5–18, calibrate on 19–22, test on 29–34.

| | Value |
|---|---|
| ERP rows generated | 1,105,386 |
| PR-AUC on test weeks — model / rules only / "attendance below 75%" | 0.565 / 0.418 / 0.206 |
| Weekly list: share of listed students on an injected trajectory (model priority / rules only) | 0.574 / 0.636 |
| Weekly list: share of listed students who declined within 6 weeks (model priority / rules only) | 0.486 / 0.403 |
| Injected trajectories listed within 4 weeks of onset | 70.6% (single-subject difficulty: 31.1%) |
| Median lead time before the term-end review | 8 weeks |
| Reason texts whose every number exists in the fact list | 26,047 of 26,047 |
| One weekly scoring and ranking pass, 1,200 students | 0.05 s on the build machine |

The weak spot is single-subject difficulty: score decline needs five assessments in a subject, so it fires late in the year.

## Scope of this prototype

Built: synthetic generator; canonical tables and read-only connector; 41 features; 12 signals; data-quality gate; model and evaluation report reproducible by one command; weekly list with reasons; next step from the playbook; parent-note drafts in English and Hindi with a number check; accept / edit / dismiss with snooze and monitoring; four-week outcome view; section alerts for the academic head; three roles (demo header identity); Docker files; tests.

Not yet built: adapter for a live ERP; PostgreSQL with one schema per school (the prototype uses SQLite); single sign-on; real message sending; React front end (the panel is plain HTML and JavaScript); multi-school admin console.

## Privacy

No real data is used. The generator has no caste, religion, income, gender or health fields. Reason text is rendered from structured facts with a `{student}` placeholder; names are merged locally after the number check. Nothing is sent to a parent — the teacher approves, and a real deployment would hand approved notes to the ERP's own messaging module.

## License

MIT — see `LICENSE`.
