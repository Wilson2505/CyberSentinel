from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pydantic import BaseModel
from typing import Optional
from app.models.pipeline import pipeline
from app.config import config
from app.database import (
    initialise_database,
    save_analysis,
    get_all_analyses,
    get_analysis_by_id,
    get_risk_statistics,
    delete_analysis
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialise database on startup
initialise_database()

app = FastAPI(
    title=config.APP_NAME,
    version=config.APP_VERSION,
    description="AI-powered cybersecurity threat intelligence analysis system"
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


# Request Models
class AnalysisRequest(BaseModel):
    log_text: Optional[str] = None
    code_snippet: Optional[str] = None
    bulletin_text: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    models_available: list


# Routes
@app.get("/")
async def dashboard(request: Request):
    """Serve the main dashboard."""
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={"app_name": config.APP_NAME}
    )


@app.get("/history")
async def history_page(request: Request):
    """Serve the analysis history page."""
    return templates.TemplateResponse(
        request=request,
        name="history.html",
        context={"app_name": config.APP_NAME}
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version=config.APP_VERSION,
        models_available=[
            "distilbert-threat-classifier",
            "mistral-code-analyser",
            "multilingual-urgency-classifier"
        ]
    )


@app.post("/analyse")
async def analyse(request: AnalysisRequest):
    """
    Main analysis endpoint — runs inputs through the full pipeline
    and saves the result to the database.
    """
    if not any([request.log_text, request.code_snippet, request.bulletin_text]):
        raise HTTPException(
            status_code=400,
            detail="At least one input required"
        )

    logger.info("Analysis request received")

    result = pipeline.analyse(
        log_text=request.log_text,
        code_snippet=request.code_snippet,
        bulletin_text=request.bulletin_text
    )

    # Save to database
    inputs = {
        "log_text": request.log_text,
        "code_snippet": request.code_snippet,
        "bulletin_text": request.bulletin_text
    }
    record_id = save_analysis(result, inputs)
    result["id"] = record_id

    return result


@app.post("/analyse/logs")
async def analyse_logs(request: AnalysisRequest):
    """Analyse log text only — Model 1."""
    if not request.log_text:
        raise HTTPException(status_code=400, detail="log_text required")
    return pipeline.threat_classifier.analyse(request.log_text)


@app.post("/analyse/code")
async def analyse_code(request: AnalysisRequest):
    """Analyse code snippet only — Model 2."""
    if not request.code_snippet:
        raise HTTPException(status_code=400, detail="code_snippet required")
    return pipeline.code_analyser.analyse(request.code_snippet)


@app.post("/analyse/bulletin")
async def analyse_bulletin(request: AnalysisRequest):
    """Analyse threat bulletin only — Model 3."""
    if not request.bulletin_text:
        raise HTTPException(status_code=400, detail="bulletin_text required")
    return pipeline.urgency_classifier.analyse(request.bulletin_text)


@app.get("/api/history")
async def get_history(limit: int = 50):
    """Returns recent analysis history from the database."""
    analyses = get_all_analyses(limit=limit)
    return {"analyses": analyses, "total": len(analyses)}


@app.get("/api/history/{analysis_id}")
async def get_analysis(analysis_id: int):
    """Returns a single analysis by ID."""
    analysis = get_analysis_by_id(analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return analysis


@app.get("/api/statistics")
async def get_statistics():
    """Returns risk level statistics across all analyses."""
    return get_risk_statistics()


@app.delete("/api/history/{analysis_id}")
async def delete_record(analysis_id: int):
    """Deletes a single analysis record."""
    deleted = delete_analysis(analysis_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return {"deleted": True, "id": analysis_id}

class FeedbackRequest(BaseModel):
    analysis_id: int
    sus_scores: list
    scenario_ratings: dict
    comments: str = ""
    participant_background: str = ""


@app.post("/api/feedback")
async def submit_feedback(request: FeedbackRequest):
    """
    Stores user testing feedback linked to an analysis ID.
    Used for formal user evaluation documentation.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Create feedback table if it doesn't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id INTEGER,
            sus_scores TEXT,
            sus_total REAL,
            scenario_ratings TEXT,
            comments TEXT,
            participant_background TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Calculate SUS score
    scores = request.sus_scores
    if len(scores) == 10:
        converted = []
        for i, score in enumerate(scores):
            if (i + 1) % 2 == 1:
                converted.append(score - 1)
            else:
                converted.append(5 - score)
        sus_total = sum(converted) * 2.5
    else:
        sus_total = 0

    import json as json_module
    cursor.execute("""
        INSERT INTO feedback (
            analysis_id, sus_scores, sus_total,
            scenario_ratings, comments, participant_background
        ) VALUES (?, ?, ?, ?, ?, ?)
    """, (
        request.analysis_id,
        json_module.dumps(request.sus_scores),
        sus_total,
        json_module.dumps(request.scenario_ratings),
        request.comments,
        request.participant_background
    ))

    conn.commit()
    record_id = cursor.lastrowid
    conn.close()

    return {
        "id": record_id,
        "sus_score": sus_total,
        "interpretation": "Above average" if sus_total >= 68 else "Below average"
    }


@app.get("/api/feedback/summary")
async def get_feedback_summary():
    """Returns aggregated user testing results."""
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            SELECT
                COUNT(*) as total_participants,
                AVG(sus_total) as avg_sus,
                MIN(sus_total) as min_sus,
                MAX(sus_total) as max_sus
            FROM feedback
        """)
        row = cursor.fetchone()
        conn.close()

        return {
            "total_participants": row[0],
            "average_sus_score": round(row[1], 1) if row[1] else 0,
            "min_sus_score": row[2],
            "max_sus_score": row[3],
            "above_average_threshold": 68
        }
    except Exception as e:
        conn.close()
        return {"error": str(e)}