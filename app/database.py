import sqlite3
import json
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = os.getenv("DB_PATH", "cybersentinel.db")


def get_connection():
    """Get a database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def initialise_database():
    """
    Creates the database tables if they don't exist.
    Called once on application startup.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            overall_risk TEXT,
            overall_urgency TEXT,
            models_used TEXT,
            log_text TEXT,
            code_snippet TEXT,
            bulletin_text TEXT,
            threat_analysis TEXT,
            code_analysis TEXT,
            urgency_analysis TEXT,
            recommendations TEXT,
            summary TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()
    logger.info("Database initialised successfully")


def save_analysis(result: dict, inputs: dict) -> int:
    """
    Saves a pipeline analysis result to the database.

    Args:
        result: The pipeline analysis result dictionary
        inputs: The original input dictionary

    Returns:
        The ID of the saved record
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO analyses (
            timestamp,
            overall_risk,
            overall_urgency,
            models_used,
            log_text,
            code_snippet,
            bulletin_text,
            threat_analysis,
            code_analysis,
            urgency_analysis,
            recommendations,
            summary
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        result.get("timestamp"),
        result.get("overall_risk"),
        result.get("overall_urgency"),
        json.dumps(result.get("models_used", [])),
        inputs.get("log_text"),
        inputs.get("code_snippet"),
        inputs.get("bulletin_text"),
        json.dumps(result.get("threat_analysis")),
        json.dumps(result.get("code_analysis")),
        json.dumps(result.get("urgency_analysis")),
        json.dumps(result.get("recommendations", [])),
        result.get("summary")
    ))

    record_id = cursor.lastrowid
    conn.commit()
    conn.close()

    logger.info(f"Analysis saved to database with ID {record_id}")
    return record_id


def get_all_analyses(limit: int = 50) -> list:
    """
    Retrieves recent analyses from the database.

    Args:
        limit: Maximum number of records to return

    Returns:
        List of analysis records as dictionaries
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            id,
            timestamp,
            overall_risk,
            overall_urgency,
            models_used,
            summary,
            created_at
        FROM analyses
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def get_analysis_by_id(analysis_id: int) -> dict:
    """
    Retrieves a single analysis by ID.

    Args:
        analysis_id: The database ID of the analysis

    Returns:
        Full analysis record as dictionary or None
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM analyses WHERE id = ?
    """, (analysis_id,))

    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    result = dict(row)

    # Parse JSON fields back to Python objects
    for field in ["models_used", "threat_analysis",
                  "code_analysis", "urgency_analysis", "recommendations"]:
        if result.get(field):
            try:
                result[field] = json.loads(result[field])
            except json.JSONDecodeError:
                pass

    return result


def get_risk_statistics() -> dict:
    """
    Returns statistics about risk levels across all analyses.
    Useful for dashboard summary and report evidence.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            overall_risk,
            COUNT(*) as count
        FROM analyses
        GROUP BY overall_risk
        ORDER BY count DESC
    """)

    rows = cursor.fetchall()
    conn.close()

    stats = {row["overall_risk"]: row["count"] for row in rows}
    total = sum(stats.values())

    return {
        "total_analyses": total,
        "by_risk_level": stats
    }


def delete_analysis(analysis_id: int) -> bool:
    """
    Deletes a single analysis record.

    Args:
        analysis_id: The database ID to delete

    Returns:
        True if deleted successfully
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
    deleted = cursor.rowcount > 0

    conn.commit()
    conn.close()

    return deleted