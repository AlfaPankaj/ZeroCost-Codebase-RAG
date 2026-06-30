import os
import ast
import re
from pathlib import Path

class PythonASTAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.imports = []
        self.routes = []
        self.api_calls = []

    def visit_Import(self, node):
        for alias in node.names:
            self.imports.append(alias.name.split('.')[0])
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module:
            self.imports.append(node.module.split('.')[0])
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call):
                func = decorator.func
                if isinstance(func, ast.Attribute) and func.attr in ['get', 'post', 'put', 'delete', 'patch']:
                    if decorator.args and isinstance(decorator.args[0], ast.Constant):
                        self.routes.append(str(decorator.args[0].value))
        self.generic_visit(node)
        
    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute):
            if getattr(node.func.value, 'id', '') in ['requests', 'httpx', 'client'] or getattr(node.func.value, 'attr', '') == 'session':
                if node.func.attr in ['get', 'post', 'put', 'delete', 'patch']:
                    if node.args and isinstance(node.args[0], ast.Constant):
                        self.api_calls.append(str(node.args[0].value))
        self.generic_visit(node)

class PolyglotAnalyzer:
    """Regex-based fallback for languages where we don't have a native AST (Java, JS, React, SQL)."""
    def __init__(self):
        self.imports = []
        self.routes = []
        self.api_calls = []
        
    def analyze(self, content: str, ext: str):
        if ext in ['.js', '.jsx', '.ts', '.tsx']:
            self.imports.extend(re.findall(r'from\s+[\'"]([^\'"]+)[\'"]', content))
            self.imports.extend(re.findall(r'require\([\'"]([^\'"]+)[\'"]\)', content))
            self.api_calls.extend(re.findall(r'(?:fetch|axios\.(?:get|post|put|delete))\([\'"]([^\'"]+)[\'"]', content))
            
        elif ext == '.java':
            self.imports.extend(re.findall(r'import\s+([\w\.]+);', content))
            self.routes.extend(re.findall(r'@(?:Get|Post|Put|Delete|Patch)Mapping\([\'"]([^\'"]+)[\'"]\)', content))
            self.routes.extend(re.findall(r'@RequestMapping.*value\s*=\s*[\'"]([^\'"]+)[\'"]', content))
            self.api_calls.extend(re.findall(r'RestTemplate\.(?:getForObject|postForObject)\([\'"]([^\'"]+)[\'"]', content))
            
        elif ext == '.sql':
            self.imports.extend(re.findall(r'REFERENCES\s+([A-Za-z0-9_]+)', content, re.IGNORECASE)) # Links tables

def generate_codebase_map(dir_path: str) -> list[dict]:
    """
    Parses a Polyglot codebase (Python, Java, JS/TS, SQL) to build a master architecture blueprint.
    """
    print("🔍 Analyzing Universal Codebase Architecture (Polyglot)...")
    
    local_modules = set()
    file_analysis = {}
    
    ignore_dirs = {"venv", ".env", "node_modules", ".git", "__pycache__", "dist", "build", "target"}
    
    for root, dirs, files in os.walk(dir_path):
        dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith('.')]
        for file in files:
            ext = Path(file).suffix.lower()
            if ext in ['.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.sql']:
                full_path = os.path.join(root, file)
                
                module_name = file.replace(ext, '')
                local_modules.add(module_name)
                
                try:
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        
                        if ext == '.py':
                            tree = ast.parse(content, filename=file)
                            analyzer = PythonASTAnalyzer()
                            analyzer.visit(tree)
                            file_analysis[full_path] = analyzer
                        else:
                            analyzer = PolyglotAnalyzer()
                            analyzer.analyze(content, ext)
                            file_analysis[full_path] = analyzer
                except Exception:
                    pass
                    
    map_chunks = []
    
    for file_path, data in file_analysis.items():
        base_name = os.path.basename(file_path)
        
        internal_imports = []
        for imp in data.imports:
            imp_base = imp.split('.')[-1]
            if imp_base in local_modules or imp in local_modules or imp.replace('./', '').replace('../', '') in local_modules:
                internal_imports.append(imp)
                
        chunk_text = f"[CODEBASE BLUEPRINT] File: {base_name}\n"
        has_data = False
        
        if internal_imports:
            chunk_text += f"- Internally Imports/References: {', '.join(set(internal_imports))}\n"
            has_data = True
            
        if data.routes:
            chunk_text += f"- Hosts API Endpoints: {', '.join(set(data.routes))}\n"
            has_data = True
            
        if data.api_calls:
            chunk_text += f"- Makes Internal API Calls To: {', '.join(set(data.api_calls))}\n"
            has_data = True
            
        if has_data:
            map_chunks.append({
                "chunk_id": f"{base_name}_blueprint",
                "text": chunk_text.strip(),
                "metadata": {"source": "codebase_analyzer", "type": "blueprint"}
            })

    if map_chunks:
        print(f"✅ Generated {len(map_chunks)} cross-language architectural blueprints.")
    return map_chunks
