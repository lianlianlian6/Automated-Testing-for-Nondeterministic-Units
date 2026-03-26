# Automated-Testing-for-Nondeterministic-Units
## Overview

This project provides an automated pipeline for program unit testing, covering intent extraction, expert knowledge retrieval and generation, reference implementation, test case generation, mutant generation, sample collection, and statistical differential testing.

---

## Pipeline

### 1. `IntentionSummary.py` — Program Unit Intent Extraction

Extracts the intention of each program unit using a fine-tuned local model.

**Configuration:**
```python
BASE_MODEL = 'your_path/your_model'   # Path to your base model
ADAPTER_PATH = 'your_path'            # Path to your LoRA adapter
JSON_PATH = './program.json'          # Path to your input JSON file
```

---

### 2. `ExpertKnowledgeRetrieval.py` — Expert Knowledge Retrieval

Retrieves domain-specific knowledge from a local PDF knowledge base to enrich code information.

**Configuration:**
- Set your model path in the script.
- Prepare a folder containing PDF files as your knowledge base.

**Usage:**
```bash
python ExpertKnowledgeRetrieval.py --folder_path /path/to/your/pdf_folder
```

> `--folder_path` should point to a directory containing PDF files.

---

### 3. `ExpertKnowledgeGeneration.py` — Expert Knowledge Analysis & Pseudocode Generation

Analyzes expert knowledge, supplements it, and converts it into algorithm pseudocode. If you do not have a local PDF knowledge base, you can skip Step 2 and start directly from this step — the model will analyze the domain knowledge on its own.

**Configuration:**
```python
openai.api_key = 'your_api_key'

input_json = "./process/program.json"   # Path to your input JSON file
output_json = "./process/program.json"  # Path to your output JSON file
```

---

### 4. `ReferenceGeneration.py` — Reference Implementation Generation

Generates reference implementations for the program units under test.

**Configuration:**
- Set your OpenAI API key.
- Set the input file path pointing to your program JSON file.

---

### 5. `TestcaseGeneration.py` — Test Case Generation

Generates test cases for the program units.

**Configuration:**
```python
openai.api_key = 'your-api-key'

input_json = r"your_path\function_dataset.json"   # Path to your input JSON file
output_json = r"your_path\function_dataset.json"  # Path to your output JSON file
start_index = 0                                   # Index to start from
```

---

### 6. `MutantGeneration.py` — Static Defect Injection & Equivalent Mutant Generation

A self-implemented static mutant generator. Supports multiple mutation types and equivalent variant generation.

**Supported mutation types:**

| Type | Default Weight |
|---|---|
| `OperatorReplacement` | 0.2 |
| `BooleanReplacement` | 0.2 |
| `CompareReplacement` | 0.2 |
| `ConstantReplacement` | 0.2 |
| `FunctionCallArgMutation` | 0.2 |

**Configuration:**
```python
openai.api_key = 'your-api-key'

# Mutant generation
input_json_path  = r"your_path\mutant_dataset.json"   # Path to your input JSON file
output_json_path = r"your_path\mutant_dataset.json"   # Path to your output JSON file
mutation_ratio   = None                               # Set a float (e.g. 0.5) to limit mutation ratio, or None for no limit

# Equivalent mutant generation
input_file  = r"your_path\mutant_dataset.json"        # Path to your input JSON file
output_file = r"your_path\mutant_dataset.json"        # Path to your output JSON file
start_index = 0                                       # Index to start from
```

---

### 7. `SampleCollection.py` — Sample Collection via Repeated Execution

Runs the program units under test multiple times to collect execution results.

**Configuration:**
```python
input_json  = r"your_path\mutant_dataset_acc.json"   # Path to your input JSON file
output_json = r"your_path\results_acc.json"          # Path to your output JSON file
repeat      = 100                                    # Number of repeated executions
```

---

### 8. `StatisticalTesting.py` — Statistical Differential Testing

Performs differential testing on collected results using statistical hypothesis testing.

**Configuration:**
```python
file1         = r"your_path\results_bm.json"                  # Results of the baseline (reference) implementation
file2         = r"your_path\results_acc.json"                  # Results of the program under test
repeat_times  = 100                                            # Should match the repeat count used in SampleCollection.py
res_json_path = r"your_path\statistical_results_acc.json"     # Path to save the statistical testing results
```
