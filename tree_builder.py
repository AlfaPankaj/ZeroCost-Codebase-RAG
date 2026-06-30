from config import CHUNK_SIZE
import requests
from config import OLLAMA_URL, CHAT_MODEL, LLM_OPTIONS

def summarize_chunks(chunks: list[dict], source: str) -> list[dict]:
    """
    Smart Filter Hierarchical Summarization (Universal RAPTOR).
    Summarizes High-Value context files across multiple languages (Python, Java, JS, Docker).
    """
    if len(chunks) < 2:
        return [] 
        
    high_value_chunks = []
    
    high_value_keywords = [
        "main.py", "__init__.py", "app.py", 
        "dockerfile", "docker-compose.yml", "package.json", 
        "app.tsx", "index.js", "main.java", "application.java"
    ]
    
    for c in chunks:
        src = c.get("metadata", {}).get("source", "").lower()
        
        if src.endswith(".md") or src.endswith(".txt"):
            high_value_chunks.append(c)
            continue
            
        for kw in high_value_keywords:
            if kw in src:
                high_value_chunks.append(c)
                break
            
    target_chunks = chunks if len(chunks) <= 50 else high_value_chunks
    
    if not target_chunks:
        print("⚡ Smart Filter: No high-level documentation or Polyglot entry points found. Skipping global tree.")
        return []
        
    print(f"🌲 Building Universal Architectural Tree (Summarizing {len(target_chunks)} High-Value chunks)...")
    
    summaries = []
    block_size = 5
    
    for i in range(0, len(target_chunks), block_size):
        block = target_chunks[i:i+block_size]
        combined_text = "\n\n".join([c["text"] for c in block])
        
        prompt = (
            "You are an expert Enterprise Architect. Write a highly detailed, comprehensive summary of the following text. "
            "Preserve all key facts, architectural details, APIs, dependencies, and concepts.\n\n"
            f"TEXT:\n{combined_text}\n\n"
            "SUMMARY:"
        )
        
        payload = {
            "model": CHAT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": LLM_OPTIONS
        }
        
        try:
            response = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=120)
            response.raise_for_status()
            summary_text = response.json().get("message", {}).get("content", "").strip()
            
            if summary_text:
                summaries.append({
                    "chunk_id": f"{source}_summary_{i//block_size}",
                    "text": f"[GLOBAL SUMMARY] {summary_text}",
                    "metadata": {"source": source, "type": "summary"}
                })
        except Exception as e:
            pass
            
    if len(summaries) > 1:
        root_summary = summarize_chunks(summaries, source + "_root")
        summaries.extend(root_summary)
        
    return summaries
