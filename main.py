import os
import hashlib

def print_header(title):
    print("\n" + "="*50)
    print(f" {title}")
    print("="*50)

def main():
    print_header("Persistent Agentic RAG - Enterprise Engine")
    print("This system will create a PERSISTENT vector mapping of a file or an entire folder.")
    print("Subsequent runs will load instantly (0.1s) unless you modify the code files.")
    
    path_input = input("\nEnter the absolute path to a file OR a folder: ").strip().strip("'\"")
    
    if not os.path.exists(path_input):
        print("Path not found! Exiting.")
        return
        
    session_id = hashlib.md5(path_input.encode()).hexdigest()[:8]
    
    from rag_agent import SessionRAGAgent
    agent = SessionRAGAgent(session_id)
    
    agent.ingest_path(path_input)

    print_header("Knowledge Mapped! Ask Questions (type 'exit' to quit)")
    
    try:
        while True:
            query = input("\n[You]: ").strip()
            if query.lower() in ['exit', 'quit']:
                break
                
            if not query:
                continue
                
            response = agent.map_reduce_ask(query)
            print(f"\n[Agent]:\n{response}")
            
    except KeyboardInterrupt:
        pass
        
    finally:
        print_header("Session Ended")
        print("Vectors and Master Synthesis caches safely saved to disk for next time.")
        print("Goodbye!")

if __name__ == "__main__":
    main()
