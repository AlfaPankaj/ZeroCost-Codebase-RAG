import os
from pathlib import Path
from config import CHUNK_SIZE, CHUNK_OVERLAP

def extract_text_from_pdf(file_path: str) -> str:
    try:
        import pymupdf4llm
        md_text = pymupdf4llm.to_markdown(file_path)
        return md_text
    except ImportError:
        print("pymupdf4llm not found. Falling back to basic pypdf.")
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            return "\n\n".join([page.extract_text() or "" for page in reader.pages])
        except ImportError:
            print("pypdf not found either. Please install pymupdf4llm.")
            return ""

def extract_text(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext in [".docx", ".pptx", ".xlsx"]:
        try:
            from markitdown import MarkItDown
            md = MarkItDown()
            result = md.convert(file_path)
            return result.text_content
        except ImportError:
            print("markitdown not found. To read Office files, please run: pip install markitdown")
            return ""
    elif ext in [".txt", ".md", ".csv", ".py", ".json", ".java", ".js", ".ts", ".jsx", ".tsx", ".sql", ".css", ".yml", ".yaml"] or os.path.basename(file_path).lower() == "dockerfile":
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return ""
    else:
        return ""

def chunk_text(text: str, source: str) -> list[dict]:
    lines = text.split('\n')
    words = []
    for line in lines:
        words.extend(line.split())
        words.append('\n')
        
    chunks = []
    idx = 0
    chunk_num = 0
    
    while idx < len(words):
        end = min(idx + CHUNK_SIZE, len(words))
        chunk_text = " ".join(words[idx:end]).replace(" \n ", "\n")
        
        if len(chunk_text.strip()) > 20:
            chunks.append({
                "chunk_id": f"{source}_chunk_{chunk_num}",
                "text": chunk_text.strip(),
                "metadata": {"source": source}
            })
            chunk_num += 1
            
        idx += (CHUNK_SIZE - CHUNK_OVERLAP)
        
    return chunks

def load_and_chunk(file_path: str) -> list[dict]:
    text = extract_text(file_path)
    if not text:
        return []
    source_name = os.path.basename(file_path)
    return chunk_text(text, source_name)

def load_directory(dir_path: str) -> list[dict]:
    """Recursively walks a directory, ignoring system/environment folders."""
    all_chunks = []
    
    supported_extensions = {
        ".pdf", ".txt", ".md", ".csv", ".py", ".json", 
        ".java", ".js", ".ts", ".jsx", ".tsx", ".sql", ".css", ".yml", ".yaml",
        ".docx", ".pptx", ".xlsx"
    }
    
    ignore_dirs = {
        "venv", ".env", "env", ".venv", "node_modules", 
        ".git", ".idea", ".vscode", "__pycache__", "dist", "build", "target", "out"
    }
    
    for root, dirs, files in os.walk(dir_path):
        dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith('.')]
        
        for file in files:
            ext = Path(file).suffix.lower()
            if ext in supported_extensions or file.lower() == "dockerfile":
                file_path = os.path.join(root, file)
                print(f"Reading: {file_path}...")
                chunks = load_and_chunk(file_path)
                all_chunks.extend(chunks)
                
    return all_chunks
