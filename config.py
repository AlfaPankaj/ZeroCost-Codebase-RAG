import os
import math

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "db_storage")

OLLAMA_URL = "http://localhost:11434"

CHAT_MODEL = "llama3.2:3b"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

LLM_OPTIONS = {
    "temperature": 0.3,
    "num_predict": 1024,
    "num_ctx": 8192, 
}

TOKENS_PER_CHUNK = int(CHUNK_SIZE * 1.3) 
MAX_SAFE_TOKENS = LLM_OPTIONS.get("num_ctx", 8192) - 1500
MAX_CHUNKS_PER_BATCH = max(1, math.floor(MAX_SAFE_TOKENS / TOKENS_PER_CHUNK))
