# Clinical Notes LLM Extractor

A local LLM-based pipeline for extracting and classifying oncology-relevant 
findings from clinical notes — including radiology reports, pathology reports, 
and medical oncologist notes — using open-source models via Ollama.

---

## Overview

This tool automates the extraction and structured classification of 
oncology-relevant tissue findings from free-text clinical notes. It is 
designed to support clinical data curation workflows where large volumes 
of unstructured notes need to be systematically reviewed for specific 
findings — a task that is time-consuming and error-prone when done manually.

The pipeline reads clinical notes from an Excel file, sends each note to a 
local LLM (Llama 3.2 via Ollama), and returns a structured JSON output 
classifying whether an oncology-relevant lesion is present and which tissue 
layer is involved.

This project was developed as part of a healthcare data science portfolio 
focused on real-world clinical data, NLP, and AI-assisted curation workflows.

---

## Clinical Context

In oncology imaging and clinical documentation, identifying whether a lesion 
involves a specific tissue layer (skin, subcutaneous, fascial, muscular, or 
soft tissue) has direct implications for:

- Staging and treatment planning
- Cohort identification in real-world data studies
- Clinical trial eligibility screening
- AI algorithm development and validation

Manual review of large volumes of CT radiology reports or oncologist notes 
for these findings is resource-intensive. This pipeline demonstrates how 
local LLMs can assist clinical data teams by providing a first-pass 
extraction layer, with human expert review as the final validation step.

---

## Key Features

- Processes any oncology-related clinical note type: CT radiology reports,
  pathology reports, medical oncologist notes, discharge summaries
- Classifies findings into 6 tissue categories + "none"
- Handles ambiguous, negated, and historical findings with explicit rules
- Structured JSON output with answer, evidence quote, and category
- Retry logic and concurrent processing for efficiency
- Resume capability: skips rows with existing results
- Fully local — no data leaves your machine (HIPAA-friendly architecture)
- Human-in-the-loop design: output is intended for expert review,
  not autonomous clinical decision-making

---

## Output Categories

| Category | Description |
|---|---|
| cutaneous/skin | Skin lesion, cutaneous mass, skin thickening |
| dermal/epidermal | Dermal or epidermal lesion or thickening |
| subcutaneous | Subcutaneous mass, nodule, lesion, or infiltration |
| fascial | Fascial thickening, enhancement, or invasion |
| muscular/intramuscular | Muscle lesion, intramuscular mass, muscle invasion |
| soft tissue | Soft tissue lesion when specific layer is not stated |
| none | No oncology-relevant finding identified |

---

## Technologies

- Python 3.10+
- Ollama (local LLM inference)
- Llama 3.2 (3B parameter model)
- Pandas
- openpyxl
- tqdm

---

## Getting Started

### Prerequisites

1. Install [Ollama](https://ollama.com/) on your local machine
2. Pull the model: `ollama pull llama3.2:3b`
3. Install Python dependencies:

```bash
pip install -r requirements.txt
```

### Input Format

Prepare an Excel file (.xlsx) with at least one column named `report` 
containing the clinical note text. One row per note.

| report |
|---|
| CT chest: There is a 1.2 cm subcutaneous nodule in the left chest wall... |
| No evidence of cutaneous or subcutaneous involvement... |

### Running the Extractor

```bash
python src/extractor.py
```

Update the `excel_path` variable in the script to point to your input file 
before running.

### Output Format

The tool appends three columns to your Excel file and saves a new file 
with the suffix `_processed_llama.xlsx`:

| Column | Description |
|---|---|
| outcome | Raw JSON response from the model |
| answer | YES or NO |
| evidence | Exact quote from the note supporting the answer |
| category | Tissue category (see table above) |

---

## Example

See the `examples/` folder for:
- `sample_notes.xlsx` — synthetic oncology clinical notes
- `sample_output.xlsx` — expected output from the extractor

---

## Important Disclaimers

- **Not for clinical use.** This tool is intended for research and data 
  curation support only. All outputs require expert human review before 
  use in any clinical or regulatory context.
- **No real patient data is included** in this repository. Example files 
  contain synthetic or publicly available deidentified notes only.
- **Local processing only.** Notes are processed entirely on your local 
  machine via Ollama and are never sent to external APIs.

---

## Portfolio Context

This project is part of a broader healthcare data science portfolio 
demonstrating applied skills in:

- Clinical NLP and LLM-based information extraction
- Real-world data curation workflows
- Python-based healthcare analytics
- Prompt engineering for structured clinical output

Other projects in this portfolio:
- [MIMIC Clinical Cohort Explorer](https://github.com/luizperandini/mimic-clinical-cohort-explorer)

---

## Author

**Luiz Perandini, PhD**
Senior Clinical Project Manager | Clinical Data Science
[LinkedIn](https://www.linkedin.com/in/luiz-perandini) | 
[GitHub](https://github.com/luizperandini)