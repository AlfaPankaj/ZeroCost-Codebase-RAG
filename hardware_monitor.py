import subprocess
import os
import json

def get_optimal_workers() -> int:
    """
    Dynamically calculates how many parallel LLM calls the current machine can handle.
    It runs only once and saves the result to a config file to save time.
    """
    config_file = "hardware_profile.json"
    
    if os.path.exists(config_file):
        try:
            with open(config_file, "r") as f:
                data = json.load(f)
                return data.get("optimal_workers", 1)
        except Exception:
            pass

    print("🖥️ [Hardware Profiler]: Running first-time hardware analysis...")
    optimal_workers = 1 
    
    try:
        vram_output = subprocess.check_output(
            ["wmic", "path", "win32_VideoController", "get", "AdapterRAM"], 
            text=True, stderr=subprocess.STDOUT
        )
        
        vram_bytes = [int(x) for x in vram_output.split() if x.isdigit()]
        
        if vram_bytes:
            max_vram_gb = max(vram_bytes) / (1024**3)
            print(f"🖥️ [Hardware Profiler]: Detected ~{max_vram_gb:.1f} GB of VRAM.")
            
            if max_vram_gb >= 12:
                optimal_workers = 4
            elif max_vram_gb >= 6:
                optimal_workers = 2
            else:
                optimal_workers = 1
        else:
            print("🖥️ [Hardware Profiler]: VRAM detection unclear. Falling back to safe CPU mode.")
            
    except Exception as e:
        print(f"🖥️ [Hardware Profiler]: Could not detect VRAM ({e}). Defaulting to 1 worker.")
        
    print(f"⚡ Setting Dynamic Parallel Limit to: {optimal_workers} concurrent LLM calls.")
    
    with open(config_file, "w") as f:
        json.dump({"optimal_workers": optimal_workers}, f)
        
    return optimal_workers
