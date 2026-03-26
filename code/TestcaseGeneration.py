import json
import time
import openai
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

openai.api_key = 'your-api-key'

CHECKPOINT_INTERVAL = 10  # save results every N functions

def extract_function_code(text):
    match = re.search(r"```python(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return ""

def save_checkpoint(functions, output_file):
    """Write current results to JSON file."""
    functions_sorted = sorted(functions, key=lambda x: x.get("id", 0))
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(functions_sorted, f, indent=4, ensure_ascii=False)
    print(f"[Checkpoint] Saved results to {output_file}")


def analyze_input(func_id, code, retries=5, delay=5):
    prompt = (
        "You are an expert in software testing.\n"
        "Given the following Python function definition, analyze its control flow, including if-else conditions, loops, and exception handling.\n"
        "Generate a list of valid, executable input examples that together maximize the function's branch coverage.\n"
        "Ensure all inputs are syntactically correct and respect the expected data types and value constraints inferred from the code.\n"
        "Only output the function call examples (e.g., 'function_name(parameter1, parameter2)') inside one Python code block without including the function code.\n"
        "List each test case on a separate line.\n"
        "Do not include any explanations or comments.\n\n"
        f"Function Code:\n{code}\n\n"
        "Output:"
    )

    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            content = response.choices[0].message["content"].strip()
            input_calls = extract_function_code(content)
            return func_id, input_calls
        except Exception as e:
            print(f"[Function {func_id}] Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                print(f"[Function {func_id}] All retries failed, leaving function empty")
                return func_id, ""

def input_generation(input_file, output_file, start_index=0):
    with open(input_file, "r", encoding="utf-8") as f:
        functions = json.load(f)

    results = []
    processed_count = 0

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_func = {
            executor.submit(analyze_input, func["id"], func["function"]): func
            for func in functions
        }

        for future in as_completed(future_to_func):
            func = future_to_func[future]
            func_id = func["id"]
            try:
                fid, input_calls = future.result()
                if input_calls.strip():  
                    results.append({
                        "id": fid,
                        "source": func.get("source", ""),
                        "code": func.get("code", ""),
                        "function": func.get("function", ""),
                        "intention": func.get("intention", ""),
                        "buggy_code": func.get("buggy_code", ""),
                        "input": input_calls
                    })
                    print(f"[Done] Function {fid} processed.")
            except Exception as e:
                print(f"[Error] Function {func_id}: {e}")
            finally:
                processed_count += 1

                # Checkpoint save
                if processed_count % CHECKPOINT_INTERVAL == 0:
                    save_checkpoint(results, output_file)

    # Final save
    save_checkpoint(results, output_file)
    print(f"Analysis finished, results written to {output_file}")

if __name__ == "__main__":
    input_json = r"your_path\function_dataset.json"
    output_json = r"your_path\function_dataset.json"

    input_generation(input_json, output_json, start_index=0)
