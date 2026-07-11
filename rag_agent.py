import requests
import json
import re
import math
import concurrent.futures
from config import OLLAMA_URL, CHAT_MODEL, LLM_OPTIONS, MAX_CHUNKS_PER_BATCH
from vector_store import PersistentVectorStore
import os
import hashlib
from hardware_monitor import get_optimal_workers

class SessionRAGAgent:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.store = PersistentVectorStore(session_id)
        self.max_parallel = get_optimal_workers()
        
    def _call_llm(self, prompt: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": CHAT_MODEL,
            "messages": messages,
            "stream": False,
            "options": LLM_OPTIONS
        }
        
        try:
            response = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=120)
            response.raise_for_status()
            return response.json().get("message", {}).get("content", "").strip()
        except Exception as e:
            print(f"[LLM Error]: {e}")
            return ""

    def ingest_path(self, target_path: str):
        from document_loader import load_and_chunk
        from tree_builder import summarize_chunks
        from codebase_analyzer import generate_codebase_map
        from pathlib import Path
        
        target_path = os.path.abspath(target_path)
        
        if os.path.isdir(target_path):
            print(f"Loading directory: {target_path}...")
            
            supported_exts = {".pdf", ".txt", ".md", ".csv", ".py", ".json", ".java", ".js", ".ts", ".jsx", ".tsx", ".sql", ".css", ".yml", ".yaml", ".docx", ".pptx", ".xlsx"}
            ignore_dirs = {"venv", ".env", "node_modules", ".git", "__pycache__", "dist", "build", "target"}
            
            files_to_process = []
            for root, dirs, files in os.walk(target_path):
                dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith('.')]
                for file in files:
                    ext = Path(file).suffix.lower()
                    if ext in supported_exts or file.lower() == "dockerfile":
                        files_to_process.append(os.path.join(root, file))
                        
            updated_files = []
            for file_path in files_to_process:
                mtime = os.path.getmtime(file_path)
                source_name = os.path.basename(file_path)
                if source_name not in self.store.file_mtimes or self.store.file_mtimes[source_name] < mtime:
                    updated_files.append((file_path, source_name, mtime))
                    
            if not updated_files:
                print("⚡ Delta Update: No files have changed since last run. Loading instantly.")
                return
                
            print(f"🔄 Delta Update: Found {len(updated_files)} new/modified files. Processing...")
            
            all_chunks = []
            for file_path, source_name, mtime in updated_files:
                if source_name in self.store.file_mtimes:
                    self.store.remove_source(source_name)
                    self.store.remove_dependent_syntheses(source_name)
                    
                chunks = load_and_chunk(file_path)
                if chunks:
                    all_chunks.extend(chunks)
                    self.store.add_chunks(chunks, source_name, mtime)
                    
            blueprints = generate_codebase_map(target_path)
            if blueprints:
                self.store.add_chunks(blueprints, "codebase_blueprints", 0)
            
            if all_chunks:
                source_name = os.path.basename(target_path.rstrip(r"\/")) + "_folder"
                summaries = summarize_chunks(all_chunks, source_name)
                self.store.add_chunks(summaries, "global_summaries", 0)

        else:
            mtime = os.path.getmtime(target_path)
            source_name = os.path.basename(target_path)
            
            if source_name in self.store.file_mtimes and self.store.file_mtimes[source_name] >= mtime:
                print("⚡ Delta Update: File hasn't changed. Loading instantly.")
                return
                
            print(f"Loading single file: {target_path}...")
            self.store.remove_source(source_name)
            self.store.remove_dependent_syntheses(source_name)
            chunks = load_and_chunk(target_path)
            if chunks:
                summaries = summarize_chunks(chunks, source_name)
                self.store.add_chunks(chunks + summaries, source_name, mtime)
            
    def expand_query(self, query: str) -> list[str]:
        prompt = (f"Generate 3 different short search queries for: '{query}'. Output ONLY queries separated by newlines.")
        expanded = self._call_llm(prompt)
        queries = [q.strip() for q in expanded.split('\n') if q.strip()]
        if query not in queries:
            queries.insert(0, query)
        return queries[:4]

    def _mini_summarize_batch(self, batch_chunks, query):
        combined_text = "\n\n".join([c["text"] for c in batch_chunks])
        prompt = (
            f"You are evaluating chunks of code/text to answer the query: '{query}'.\n"
            "Extract any relevant facts, logic, or dependencies from the text below.\n"
            "If it is irrelevant, output 'NOT RELEVANT'.\n\n"
            f"TEXT:\n{combined_text}"
        )
        return self._call_llm(prompt)

    def map_reduce_ask(self, query: str) -> str:
        if not self.store.ids:
            return "Memory empty. Please map a file or folder first."
            
        print("\n🧠 [Agent]: Expanding query for massive Map-Reduce retrieval...")
        search_queries = self.expand_query(query)
        
        all_results = []
        for q in search_queries:
            all_results.extend(self.store.search(q, 5))
            
        seen = set()
        unique_chunks = []
        dependent_files = set()
        for r in all_results:
            if r['text'] not in seen:
                seen.add(r['text'])
                unique_chunks.append(r)
                if "source" in r["metadata"] and not r["metadata"]["source"].endswith("_folder") and r["metadata"]["source"] != "map_reduce_cache":
                    dependent_files.add(r["metadata"]["source"])
                
        ideal_batch_size = max(1, math.ceil(len(unique_chunks) / self.max_parallel))
        
        batch_size = min(ideal_batch_size, MAX_CHUNKS_PER_BATCH)
        
        batches = [unique_chunks[i:i + batch_size] for i in range(0, len(unique_chunks), batch_size)]
        
        print(f"🔄 [Dynamic Math]: Safe limit is {MAX_CHUNKS_PER_BATCH} chunks per prompt.")
        print(f"🔄 [Dynamic Math]: Split {len(unique_chunks)} chunks into {len(batches)} optimal batches (Size: {batch_size}).")
        print(f"⚡ [Concurrency]: Utilizing {self.max_parallel} parallel workers based on your hardware profile...")
        
        ephemeral_summaries = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            futures = {executor.submit(self._mini_summarize_batch, batch, query): i for i, batch in enumerate(batches)}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if "NOT RELEVANT" not in result.upper():
                    ephemeral_summaries.append(result)
                    
        print("🧠 [Synthesis]: Combining intermediate ephemeral memory into Master Answer...")
        
        if not ephemeral_summaries:
            return "I could not find any highly relevant information in the codebase to answer that specific query. The AI workers deemed the retrieved code chunks as not relevant. Could you please rephrase or be more specific?"
            
        combined_ephemeral = "\n\n--- Intermediate Summary ---\n".join(ephemeral_summaries)
        
        final_prompt = (
            f"You are the Master AI Synthesizer. Based on the following intermediate summaries, "
            f"provide a highly detailed and perfect answer to the user's question.\n\n"
            f"USER QUESTION: {query}\n\n"
            f"INTERMEDIATE SUMMARIES:\n{combined_ephemeral}\n\n"
            "FINAL ANSWER:"
        )
        
        master_answer = self._call_llm(final_prompt)
        
        print("💾 [Smart Cache]: Saving Master Synthesis to local DB (with dependency tracking).")
        synthesis_chunk = [{
            "chunk_id": f"synthesis_{hashlib.md5(query.encode()).hexdigest()[:8]}",
            "text": f"[CACHED MASTER ANSWER for '{query}']\n{master_answer}",
            "metadata": {
                "source": "map_reduce_cache", 
                "type": "synthesis",
                "dependent_files": list(dependent_files)
            }
        }]
        
        self.store.add_chunks(synthesis_chunk, "map_reduce_cache", 0)
        return master_answer

    def cleanup(self):
        pass
