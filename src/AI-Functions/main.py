import os
import sys
import json
import inspect
import importlib.util
from pathlib import Path
import base64
from google import genai
from google.genai import types

# Add parent directory to path to find AI_Function module
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class AI_Functions:
    def __init__(self):
        self.function_map = {}
        self.class_instances = {}
        self._discover_and_load_functions()
    
    def _discover_and_load_functions(self):
        """Dynamically discover and load all classes and functions from AI_*.py files"""
        ai_function_dir = Path(__file__).parent.parent.parent / "AI_Function"
        
        if not ai_function_dir.exists():
            print(f"Warning: AI_Function directory not found at {ai_function_dir}")
            return
        
        # Find all AI_*.py files
        ai_files = list(ai_function_dir.glob("AI_*.py"))
        
        for file_path in ai_files:
            try:
                # Load the module dynamically
                spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Find all classes in the module
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if obj.__module__ == module.__name__:  # Only classes defined in this module
                        try:
                            # Instantiate the class and get its methods
                            instance = obj()
                            self.class_instances[name] = instance
                            
                            # Add all public methods to function_map
                            for method_name, method in inspect.getmembers(instance, inspect.ismethod):
                                if not method_name.startswith("_"):
                                    # Create a unique key for the function
                                    func_key = f"{name.lower()}_{method_name}".replace("calculator_", "")
                                    self.function_map[func_key] = method
                                    # Also add without class name for convenience
                                    self.function_map[method_name] = method
                        except Exception as e:
                            print(f"Warning: Could not instantiate class {name} from {file_path.name}: {e}")
                
                # Find all functions in the module
                for name, obj in inspect.getmembers(module, inspect.isfunction):
                    if obj.__module__ == module.__name__:  # Only functions defined in this module
                        self.function_map[name] = obj
                
            except Exception as e:
                print(f"Warning: Could not load {file_path.name}: {e}")

    def process_query(self, user_query, key="AIzaSyBYFZveb5j2EY6tbmSE6Bkyugi6pA8GPOk", model="gemini-2.5-flash"):
        client = genai.Client(api_key=key)
        
        # Get available functions for the AI to use
        available_functions = ", ".join(self.function_map.keys())
        
        # Ask AI to identify the operation and extract parameters
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=f"Parse this query and respond with ONLY valid JSON (no markdown): {user_query}\n"
                        f"Available functions: {available_functions}\n"
                        "Return format: {{\"function\": \"function_name\", \"args\": [arg1, arg2, ...]}}"
                    ),
                ],
            ),
        ]
        
        response_text = ""
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
        ):
            response_text += chunk.text
        
        # Parse AI response and execute function
        try:
            # Clean up response (remove markdown code blocks if present)
            response_text = response_text.replace("```json", "").replace("```", "").strip()
            result_data = json.loads(response_text)
            print(f"Raw AI response: {response_text}")
            print(f"Parsed result_data: {result_data}")

            function_name = result_data.get("function")
            args = result_data.get("args", [])
            
            if function_name in self.function_map:
                result = self.function_map[function_name](*args)
                return f"Function: {function_name}, Arguments: {args}, Result: {result}"
            else:
                return f"Function '{function_name}' not found. Available functions: {', '.join(self.function_map.keys())}"
        except json.JSONDecodeError:
            return f"Failed to parse AI response: {response_text}"


# Example usage
if __name__ == "__main__":
    ai_func = AI_Functions()
    
    # Test queries
    queries = [
        "What is 10 plus 2?",
        "Can you lowercase the string 'HELLO WORLD'?",
        "Calculate 5 to the power of 3.",
    ]
    
    for query in queries:
        print(f"Query: {query}")
        print(f"Result: {ai_func.process_query(query)}\n")