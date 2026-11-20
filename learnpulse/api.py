"""FastAPI service and the teacher panel.

Identity in the prototype is a demo header, `X-Demo-User`:
`teacher:<class>`, `head` or `counsellor`. The service enforces what each
role can see and do. Single sign-on is out of scope for the prototype.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .service import LearnPulseService, User

STATIC = Path(__file__).parent / "static"


class DecisionIn(BaseModel):
    student_id: str
    week: int
    action: Literal["accept", "edit", "dismiss"]
    playbook_key: str | None = None
    step: str | None = None
    note: str | None = None
    dismiss_reason: str | None = None


def create_app(db: Path, report_dir: Path = Path("reports"), rewriter=None) -> FastAPI:
    svc = LearnPulseService(db, rewriter=rewriter)
    app = FastAPI(title="LearnPulse API", version="0.3.0",
                  description="Early-warning and intervention copilot for school ERPs (prototype, synthetic data).")
    app.state.svc = svc

    def user(x_demo_user: str | None = Header(default=None)) -> User:
        try:
            return User.parse(x_demo_user)
        except PermissionError as e:
            raise HTTPException(401, str(e))

    def guard(fn, *a):
        try:
            return fn(*a)
        except PermissionError as e:
            raise HTTPException(403, str(e))
        except KeyError as e:
            raise HTTPException(404, f"not found: {e}")
        except ValueError as e:
            raise HTTPException(422, str(e))

    @app.get("/api/health")
    def health():
        return {"status": "ok", "students": svc.A.n, "classes": len(svc.classes)}

    @app.get("/api/meta")
    def meta():
        return svc.meta()

    @app.get("/api/list")
    def weekly_list(klass: str, week: int, u: User = Depends(user)):
        return guard(svc.weekly_list, u, klass, week)

    @app.get("/api/students/{sid}")
    def student(sid: str, week: int, u: User = Depends(user)):
        return guard(svc.student, u, sid, week)

    @app.post("/api/decisions")
    def decide(body: DecisionIn, u: User = Depends(user)):
        return guard(svc.decide, u, body.model_dump())

    @app.get("/api/outcomes")
    def outcomes(klass: str | None = None, u: User = Depends(user)):
        return guard(svc.outcomes, u, klass)

    @app.get("/api/section-alerts")
    def section_alerts(week: int, u: User = Depends(user)):
        return guard(svc.section_alerts, u, week)

    @app.post("/api/demo/reset")
    def reset(u: User = Depends(user)):
        guard(svc.reset, u)
        return {"status": "reset"}

    @app.get("/api/evaluation")
    def evaluation():
        f = report_dir / "evaluation.json"
        if not f.exists():
            raise HTTPException(404, "run `python -m learnpulse build` first")
        return JSONResponse(json.loads(f.read_text(encoding="utf-8")))

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    return app
