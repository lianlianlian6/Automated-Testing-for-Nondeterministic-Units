import ast
import random
import copy
import json
from typing import List, Dict
import json
import time
import openai
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import math

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

# ==================== Mutator ====================
class OperatorMutator(ast.NodeTransformer):
    OPS = {
        ast.Add: ast.Sub, ast.Sub: ast.Add,
        ast.Mult: ast.Div, ast.Div: ast.Mult,
        ast.Mod: ast.FloorDiv, ast.FloorDiv: ast.Mod,
        ast.Pow: ast.Mult
    }

    def __init__(self, num_mutations=1):
        self.num_mutations = num_mutations
        self.mutated_count = 0

    def visit_BinOp(self, node):
        self.generic_visit(node)
        if self.mutated_count < self.num_mutations and type(node.op) in self.OPS:
            node.op = self.OPS[type(node.op)]()
            self.mutated_count += 1
        return node


class BooleanMutator(ast.NodeTransformer):
    def __init__(self, num_mutations=1):
        self.num_mutations = num_mutations
        self.mutated_count = 0

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        if self.mutated_count < self.num_mutations:
            if isinstance(node.op, ast.And):
                node.op = ast.Or()
            elif isinstance(node.op, ast.Or):
                node.op = ast.And()
            self.mutated_count += 1
        return node


class CompareMutator(ast.NodeTransformer):
    MAP = {
        ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
        ast.Lt: ast.Gt, ast.Gt: ast.Lt,
        ast.LtE: ast.GtE, ast.GtE: ast.LtE
    }

    def __init__(self, num_mutations=1):
        self.num_mutations = num_mutations
        self.mutated_count = 0

    def visit_Compare(self, node):
        self.generic_visit(node)
        for i, op in enumerate(node.ops):
            if self.mutated_count >= self.num_mutations:
                break
            if type(op) in self.MAP:
                node.ops[i] = self.MAP[type(op)]()
                self.mutated_count += 1
        return node


class ConstantMutator(ast.NodeTransformer):
    def __init__(self, num_mutations=1):
        self.num_mutations = num_mutations
        self.mutated_count = 0

    def visit_Constant(self, node):
        if self.mutated_count < self.num_mutations and isinstance(node.value, (int, float)):
            self.mutated_count += 1
            return ast.Constant(value=node.value + random.choice([-1, 1, 2]))
        return node


class FunctionCallArgMutator(ast.NodeTransformer):
    def __init__(self, num_mutations=1):
        self.num_mutations = num_mutations
        self.mutated_count = 0
        self.operators = ["permute", "delete", "replace_const"]

    def visit_Call(self, node):
        self.generic_visit(node)
        if self.mutated_count >= self.num_mutations or not node.args:
            return node

        op = random.choice(self.operators)
        new_node = copy.deepcopy(node)

        try:
            if op == "permute" and len(new_node.args) > 1:
                random.shuffle(new_node.args)
            elif op == "delete" and len(new_node.args) > 1:
                del new_node.args[random.randrange(len(new_node.args))]
            elif op == "replace_const":
                idx = random.randrange(len(new_node.args))
                new_node.args[idx] = ast.Constant(value=random.choice([0, 1, -1, 42]))

            self.mutated_count += 1
            return new_node
        except Exception:
            return node

class StaticMutator:
    def __init__(self, mutation_weights: Dict[str, float]):
        self.mutators = {
            "OperatorReplacement": OperatorMutator,
            "BooleanReplacement": BooleanMutator,
            "CompareReplacement": CompareMutator,
            "ConstantReplacement": ConstantMutator,
            "FunctionCallArgMutation": FunctionCallArgMutator
        }
        self.weights = mutation_weights

    def mutate_once(self, code: str, exclude_type=None, num_mutations: int = 1):
        mutation_types = list(self.mutators.keys())
        weights = [self.weights[t] for t in mutation_types]

        if exclude_type and exclude_type in mutation_types:
            idx = mutation_types.index(exclude_type)
            mutation_types.pop(idx)
            weights.pop(idx)

        mutation_type = random.choices(mutation_types, weights=weights, k=1)[0]

        mutator_class = self.mutators[mutation_type]
        mutator = mutator_class(num_mutations=num_mutations)

        tree = ast.parse(code)
        mutated_tree = copy.deepcopy(tree)
        mutated_tree = mutator.visit(mutated_tree)
        ast.fix_missing_locations(mutated_tree)
        mutated_code = ast.unparse(mutated_tree)

        return mutation_type, mutated_code


def remove_comments_from_code(code: str) -> str:
    class CommentRemover(ast.NodeTransformer):
        def visit(self, node):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if (len(node.body) > 0 and isinstance(node.body[0], ast.Expr)
                        and isinstance(getattr(node.body[0], 'value', None), ast.Constant)
                        and isinstance(node.body[0].value.value, str)):
                    node.body.pop(0)
            return self.generic_visit(node)

    tree = ast.parse(code)
    tree = CommentRemover().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def generate_valid_mutation(mutator, base_func, exclude_type=None, max_retry=5, prev_code=None, num_mutations=1):
    available_types = list(mutator.mutators.keys())
    random.shuffle(available_types)

    if exclude_type:
        available_types = [t for t in available_types if t not in exclude_type]

    for mtype in available_types:
        for _ in range(max_retry):
            try:
                mtype_used, code = mutator.mutate_once(base_func, exclude_type=exclude_type, num_mutations=num_mutations)
                code_no_comment = remove_comments_from_code(code)
                base_no_comment = remove_comments_from_code(base_func)
                if code_no_comment != base_no_comment and code_no_comment != prev_code:
                    return mtype_used, code_no_comment
            except Exception:
                continue
    return None, None

def generate_mutants_for_json(input_json_path: str, output_json_path: str,
                              mutation_weights: Dict[str, float],
                              mutants_per_sample=2,
                              mutation_ratio: float = None,  
                              num_nodes: int = 1,
                              keep_fields: List[str] = None):
    if keep_fields is None:
        keep_fields = ["id", "source", "function", "test_input"]

    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mutator = StaticMutator(mutation_weights)
    output_data = []

    for item in data:
        filtered = {k: item.get(k) for k in keep_fields if k in item}
        base_func = filtered.get("function")
        if not base_func:
            continue
        try:
            tree = ast.parse(base_func)
            total_nodes = sum(1 for _ in ast.walk(tree))
        except Exception as e:
            print(f"❌ Failed on（id={item.get('id')}）：{e}")
            continue

        if mutation_ratio is not None:
            num_mut = max(1, math.floor(total_nodes * mutation_ratio))
        else:
            num_mut = num_nodes

        mutants = []
        used_types = set()
        prev_code = None

        for _ in range(mutants_per_sample):
            mutation_type, mutated_code = generate_valid_mutation(
                mutator, base_func, max_retry=5, exclude_type=used_types,
                prev_code=prev_code,
                num_mutations=num_mut
            )
            if mutation_type and mutated_code:
                mutants.append({
                    "mutation_type": mutation_type,
                    "mutated_function": mutated_code
                })
                used_types.add(mutation_type)
                prev_code = mutated_code

        if mutants:
            filtered["function"] = remove_comments_from_code(base_func)
            filtered["mutants"] = mutants
            # filtered["mutation_nodes"] = num_mut  
            # filtered["total_nodes"] = total_nodes  
            # filtered["mutation_ratio"] = round(num_mut / total_nodes, 4)  
            output_data.append(filtered)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print(f"✅ Success：{output_json_path}")


def equal_code(func_id, code, n: int = 2, retries=5, delay=5):
    prompt = f"""
    You are an expert Python developer and code refactoring specialist. You are given the following function:
    {code}
    
    Your task is as follows:
    1. Analyze the purpose and logic of the given function.
    2. Generate {n} functionally equivalent variants of this function.
        - The variants must preserve the original behavior: given the same test inputs, the outputs must be identical to the original function.
        - Function names must remain exactly the same.
        - Keep all import statements intact.
        - Only modify internal implementation details while preserving the logic.
    3. Output strictly valid JSON in the format:
        {{
          "mutants": [
            {{"mutation_type": "Equivalent", "mutated_function": "<variant1_code>"}},
            {{"mutation_type": "Equivalent", "mutated_function": "<variant2_code>"}}
          ]
        }}
    4. The generated function must still be executable without syntax errors or crashes on simple inputs.
        - Do NOT include any file I/O (no reading/writing files).
        - Do NOT include user interaction (no input, no print).
        - Do NOT include any demo in `if __name__ == "__main__":`.
    
    Instructions:
    - Only return the JSON with "mutants".
    - Do not include any extra explanation, comments, or text outside the JSON.
    """

    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            content = response.choices[0].message["content"].strip()
            data = json.loads(content)
            mutants = data.get("mutants", [])
            return func_id, mutants
        except Exception as e:
            print(f"[Function {func_id}] Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                print(f"[Function {func_id}] All retries failed, leaving function empty")
                return func_id, ""

def equal_generation(input_file, output_file, start_index=0):
    with open(input_file, "r", encoding="utf-8") as f:
        functions = json.load(f)

    results = []
    processed_count = 0

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_func = {
            executor.submit(equal_code, func["id"], func["function"]): func
            for func in functions[start_index:]
        }

        for future in as_completed(future_to_func):
            func = future_to_func[future]
            func_id = func["id"]
            try:
                fid, mutants_generated = future.result()
                if mutants_generated:  
                    if "mutants" in func and func["mutants"]:
                        func["mutants"].extend(mutants_generated)
                    else:
                        func["mutants"] = mutants_generated

                    results.append(func)
                    print(f"[Done] Function {fid} processed with {len(mutants_generated)} equivalent mutants.")
                else:
                    results.append(func)
                    print(f"[Skipped] Function {fid} no equivalent mutants generated.")
            except Exception as e:
                print(f"[Error] Function {func_id}: {e}")
                results.append(func)
            finally:
                processed_count += 1

                # Checkpoint save
                if processed_count % CHECKPOINT_INTERVAL == 0:
                    save_checkpoint(results, output_file)

    # Final save
    save_checkpoint(results, output_file)
    print(f"Analysis finished, results written to {output_file}")

if __name__ == "__main__":
    mutation_weights = {
        "OperatorReplacement": 0.2,
        "BooleanReplacement": 0.2,
        "CompareReplacement": 0.2,
        "ConstantReplacement": 0.2,
        "FunctionCallArgMutation": 0.2
    }

    generate_mutants_for_json(
        input_json_path= r"your_path\mutant_dataset.json",
        output_json_path= r"your_path\mutant_dataset.json",
        mutation_weights=mutation_weights,
        mutation_ratio = None,
        keep_fields=["id", "source", "function", "test_input"]
    )

    ### Equivalent mutant generation
    input_file = r"your_path\mutant_dataset.json"
    output_file = r"your_path\mutant_dataset.json"
    equal_generation(input_file, output_file, start_index=0)
