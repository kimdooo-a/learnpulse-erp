"""End to end: build the synthetic school, then drive the API as each role."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from learnpulse.api import create_app

T8B = {"X-Demo-User": "teacher:8-B"}
HEAD = {"X-Demo-User": "head"}


@pytest.fixture(scope="module")
def client(built_db):
    app = create_app(built_db, report_dir=built_db.parent / "reports")
    with TestClient(app) as c:
        yield c


def test_build_report(built_db):
    r = json.loads((built_db.parent / "reports" / "evaluation.json").read_text(encoding="utf-8"))
    assert r["dataset"]["synthetic"] is True
    assert r["dataset"]["erp_rows_total"] > 1_100_000
    assert r["text_fidelity"]["share"] == 1.0
    assert r["metrics"]["pr_auc_model"] > r["metrics"]["pr_auc_attendance_below_75"]


def test_identity_is_required_and_scoped(client):
    assert client.get("/api/list", params={"klass": "8-B", "week": 30}).status_code == 401
    assert client.get("/api/list", params={"klass": "8-C", "week": 30}, headers=T8B).status_code == 403
    assert client.get("/api/section-alerts", params={"week": 30}, headers=T8B).status_code == 403
    assert client.get("/api/section-alerts", params={"week": 30}, headers=HEAD).status_code == 200


def test_weekly_list_has_reasons_and_at_most_five(client):
    r = client.get("/api/list", params={"klass": "8-B", "week": 30}, headers=T8B).json()
    assert r["status"] == "ok"
    assert 1 <= len(r["students"]) <= 5
    pr = [s["priority"] for s in r["students"]]
    assert pr == sorted(pr, reverse=True)
    for s in r["students"]:
        assert s["reasons"], "every listed student has a stated reason"
        assert all(a["signal"] != "F1" for a in s["annotations"])  # fee context hidden from teachers


def test_decision_flow_monitoring_and_snooze(client):
    client.post("/api/demo/reset", headers=HEAD)
    first = client.get("/api/list", params={"klass": "8-B", "week": 30}, headers=T8B).json()["students"]
    a, b = first[0]["student_id"], first[1]["student_id"]
    assert client.post("/api/decisions", json={"student_id": b, "week": 30, "action": "dismiss"}, headers=T8B).status_code == 422
    assert client.post("/api/decisions", json={"student_id": a, "week": 30, "action": "accept"}, headers=T8B).status_code == 200
    assert client.post("/api/decisions", json={"student_id": b, "week": 30, "action": "dismiss",
                                               "dismiss_reason": "Known illness"}, headers=T8B).status_code == 200
    same = {s["student_id"]: s for s in client.get("/api/list", params={"klass": "8-B", "week": 30}, headers=T8B).json()["students"]}
    assert same[a]["decision"]["action"] == "accept"
    nxt = {s["student_id"] for s in client.get("/api/list", params={"klass": "8-B", "week": 31}, headers=T8B).json()["students"]}
    assert a not in nxt and b not in nxt  # monitoring / snoozed
    out = client.get("/api/outcomes", headers=T8B).json()
    st = {r["student_id"]: r["status"] for r in out["decisions"]}
    assert st[b] == "dismissed"
    assert st[a] in ("recovered", "not recovered", "pending")
    client.post("/api/demo/reset", headers=HEAD)


def test_other_teacher_cannot_decide(client):
    s = client.get("/api/list", params={"klass": "8-B", "week": 30}, headers=T8B).json()["students"][0]["student_id"]
    r = client.post("/api/decisions", json={"student_id": s, "week": 30, "action": "accept"}, headers={"X-Demo-User": "teacher:9-A"})
    assert r.status_code == 403


def test_student_detail_drafts_are_verified(client):
    s = client.get("/api/list", params={"klass": "8-B", "week": 30}, headers=T8B).json()["students"][0]["student_id"]
    d = client.get(f"/api/students/{s}", params={"week": 30}, headers=T8B).json()
    assert d["facts"] and len(d["attendance_trend"]) == 30
    assert set(d["parent_drafts"]) == {"en", "hi"}
    assert "{student}" not in d["parent_drafts"]["en"]["text"]
