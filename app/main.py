from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pydantic import BaseModel
from typing import Optional
from app.models.pipeline import pipeline
from app.config import config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=config.APP_NAME,
    version=config.APP_VERSION,
    description="AI-powered cybersecurity threat intelligence analysis system"
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


# ── Request Models ───────────────────────────────────────────────────────────

class AnalysisRequest(BaseModel):
    log_text: Optional[str] = None
    code_snippet: Optional[str] = None
    bulletin_text: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    models_available: list


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
async def dashboard(request: Request):
    """Serve the main dashboard."""
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={"app_name": config.APP_NAME}
    )


@app.get("/health")
async def health_check():
    """Health check endpoint — verifies system is running."""
    return HealthResponse(
        status="healthy",
        version=config.APP_VERSION,
        models_available=[
            "distilbert-threat-classifier",
            "codellama-code-analyser",
            "multilingual-urgency-classifier"
        ]
    )


@app.post("/analyse")
async def analyse(request: AnalysisRequest):
    """
    Main analysis endpoint — runs inputs through the full pipeline.

    Accepts any combination of:
    - log_text: system logs or security reports
    - code_snippet: suspicious code to analyse
    - bulletin_text: CVE descriptions or threat advisories

    Returns unified threat assessment report.
    """
    if not any([request.log_text, request.code_snippet, request.bulletin_text]):
        raise HTTPException(
            status_code=400,
            detail="At least one input required: log_text, code_snippet, or bulletin_text"
        )

    logger.info("Analysis request received")

    result = pipeline.analyse(
        log_text=request.log_text,
        code_snippet=request.code_snippet,
        bulletin_text=request.bulletin_text
    )

    return result


@app.post("/analyse/logs")
async def analyse_logs(request: AnalysisRequest):
    """Analyse log text only — Model 1."""
    if not request.log_text:
        raise HTTPException(status_code=400, detail="log_text required")

    result = pipeline.threat_classifier.analyse(request.log_text)
    return result


@app.post("/analyse/code")
async def analyse_code(request: AnalysisRequest):
    """Analyse code snippet only — Model 2."""
    if not request.code_snippet:
        raise HTTPException(status_code=400, detail="code_snippet required")

    result = pipeline.code_analyser.analyse(request.code_snippet)
    return result


@app.post("/analyse/bulletin")
async def analyse_bulletin(request: AnalysisRequest):
    """Analyse threat bulletin only — Model 3."""
    if not request.bulletin_text:
        raise HTTPException(status_code=400, detail="bulletin_text required")

    result = pipeline.urgency_classifier.analyse(request.bulletin_text)
    return result