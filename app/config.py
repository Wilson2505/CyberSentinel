import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Ollama settings
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_CODE_MODEL = os.getenv("OLLAMA_CODE_MODEL", "codellama")
    OLLAMA_SYNTHESIS_MODEL = os.getenv("OLLAMA_SYNTHESIS_MODEL", "mistral")
    
    # HuggingFace settings
    HF_THREAT_MODEL = os.getenv(
        "HF_THREAT_MODEL", 
        "distilbert-base-uncased-finetuned-sst-2-english"
    )
    HF_URGENCY_MODEL = os.getenv(
        "HF_URGENCY_MODEL",
        "tabularisai/multilingual-sentiment-analysis"
    )
    
    # App settings
    APP_NAME = "CyberSentinel"
    APP_VERSION = "0.1.0"
    DEBUG = os.getenv("DEBUG", "True") == "True"
    
    # Model evaluation settings
    CONFIDENCE_THRESHOLD = 0.7
    
config = Config()