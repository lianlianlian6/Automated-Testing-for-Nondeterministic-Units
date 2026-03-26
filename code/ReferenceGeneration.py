import openai
import json
import os
from tqdm import tqdm
import time
import concurrent.futures
import re
 
openai.api_key = 'your_api_key'
 
def extract_python_code(text):
    match = re.search(r"```python(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return text.strip()
 
def generate_function_code(description, code, retries=5, delay=5):
    prompt = (
        "You are an expert Python programmer.\n"
        "Your task is to translate the provided algorithm pseudocode into a complete, correct, and executable Python function.\n\n"
        "Instructions:\n"
        "1. The pseudocode is the authoritative source for the algorithm's logic — follow it strictly.\n"
        "2. You must adopt the coding style, naming conventions, and structure from the provided reference code. Specifically:\n"
        "   - Use the same function name and parameter names as in the reference code.\n"
        "   - Match its formatting, indentation, and general stylistic conventions.\n"
        "3. If the reference code's logic conflicts with the pseudocode, **always follow the pseudocode**.\n"
        "   - Use the reference code only for stylistic or structural cues (not for logic correctness).\n\n"
        "Your final implementation must:\n"
        "- Include all parameters and logic from the pseudocode.\n"
        "- Use 'return' statements instead of 'print'.\n"
        "- Be complete: no placeholders (e.g., 'pass') or omitted logic.\n"
        "- Avoid test cases, main blocks, or unnecessary imports.\n\n"
        "Output only the final Python function, enclosed in a single code block using triple backticks (```python ... ```), with NO additional explanation or comments.\n\n"
        "Pseudocode:\n"
        f"{description}\n\n"
        "Reference Code (for structure/style only):\n"
        f"{code}\n\n"
        "Output:"
    )
 
    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a professional Python developer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            content = response.choices[0].message['content'].strip()
            extract_code = extract_python_code(content)
            return extract_code
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                print("All retries failed for a program.")
                return ""
 
def process_single_generation(idx_program):
    idx, program = idx_program
    description = program.get("description", "")
    code = program.get("test_program", "")
    if description:
        generated_code = generate_function_code(description, code)
        program["generation"] = generated_code
        time.sleep(1)
    return idx, program
 
 
def process_json_add_generation(input_json, output_json, num_entries=None, max_workers=5):
    with open(input_json, 'r', encoding='utf-8') as f:
        programs = json.load(f)
 
    filtered_programs = [(idx, programs[idx]) for idx in range(len(programs))]
 
    if num_entries is not None:
        selected_programs = filtered_programs[:min(num_entries, len(filtered_programs))]
    else:
        selected_programs = filtered_programs
 
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = list(tqdm(executor.map(process_single_generation, selected_programs), total=len(selected_programs), desc="Generating functions"))
 
    for idx, updated_program in futures:
        programs[idx] = updated_program
 
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(programs, f, ensure_ascii=False, indent=2)
 
if __name__ == "__main__":
    input_json = "./process/program.json"
    output_json = "./process/program.json"
 
    print("🔍 Starting code generation...")
    process_json_add_generation(input_json, output_json, num_entries=None, max_workers=5)
 
    print("✅ Reference generation completed.")
 
