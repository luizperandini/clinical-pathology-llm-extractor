"""Extract structured prostate-biopsy findings from narrative pathology reports."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional progress display
    def tqdm(iterable, **_kwargs):
        return iterable


OUTCOMES = {"positive", "negative", "atypical", "indeterminate", "unavailable"}
TRISTATE = {"present", "absent", "not_reported"}
EXTRACTION_COLUMNS = [
    "biopsy_outcome",
    "histology",
    "highest_gleason_primary",
    "highest_gleason_secondary",
    "highest_gleason_total",
    "isup_grade_group",
    "positive_cores",
    "total_cores",
    "max_tumor_involvement_percent",
    "perineural_invasion",
    "lymphovascular_invasion",
    "evidence",
    "extraction_notes",
]


def build_prompt(report_text: str) -> str:
    return f"""
You are a clinical information extraction assistant specializing in prostate pathology.

Extract findings ONLY from the prostate needle-biopsy report below. Do not infer a finding
that is not stated. Ignore surgical prostatectomy results if they appear in the text.

REPORT
\"\"\"
{report_text.strip()}
\"\"\"

Return exactly one JSON object with these keys:
{{
  "biopsy_outcome": "positive | negative | atypical | indeterminate | unavailable",
  "histology": "string or null",
  "highest_gleason_primary": "integer 1-5 or null",
  "highest_gleason_secondary": "integer 1-5 or null",
  "highest_gleason_total": "integer 2-10 or null",
  "positive_cores": "integer or null",
  "total_cores": "integer or null",
  "max_tumor_involvement_percent": "number 0-100 or null",
  "perineural_invasion": "present | absent | not_reported",
  "lymphovascular_invasion": "present | absent | not_reported",
  "evidence": "brief exact quote supporting the outcome and highest Gleason score",
  "extraction_notes": "brief ambiguity note or empty string"
}}

Rules:
- positive: invasive malignancy is identified in at least one biopsy core.
- negative: the biopsies explicitly contain no malignancy/tumor and no atypical focus.
- atypical: atypical glands, ASAP, high-grade PIN, or another non-definitive abnormality is
  reported without confirmed invasive malignancy.
- indeterminate: the text exists but does not permit classification.
- unavailable: the report explicitly says that no biopsy/report/tissue is available.
- Select the HIGHEST Gleason score in the report. For equal totals, 4+3 is higher than 3+4.
- Count positive and total cores only when they can be determined from the report. Otherwise null.
- max_tumor_involvement_percent is the highest stated percent of tissue/core involvement.
- Do not count perineural invasion as absent unless explicitly negated.
- Return JSON only, with no Markdown.
""".strip()


def extract_first_json_object(text: str) -> str:
    if not isinstance(text, str):
        return ""
    start = text.find("{")
    if start < 0:
        return text.strip()
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
        elif char == "\\" and in_string:
            escaped = True
        elif char == '"':
            in_string = not in_string
        elif not in_string:
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start:index + 1]
    return text[start:].strip()


def _nullable_int(value: Any, minimum: int, maximum: int) -> int | None:
    if value is None or value == "":
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if minimum <= number <= maximum else None


def _nullable_float(value: Any, minimum: float, maximum: float) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if minimum <= number <= maximum else None


def derive_isup_grade_group(primary: int | None, secondary: int | None, total: int | None) -> int | None:
    if total == 6 and primary == 3 and secondary == 3:
        return 1
    if total == 7 and primary == 3 and secondary == 4:
        return 2
    if total == 7 and primary == 4 and secondary == 3:
        return 3
    if total == 8:
        return 4
    if total in {9, 10}:
        return 5
    return None


def normalize_extraction(obj: dict[str, Any]) -> dict[str, Any]:
    outcome = str(obj.get("biopsy_outcome", "indeterminate")).strip().lower()
    if outcome not in OUTCOMES:
        outcome = "indeterminate"

    histology = obj.get("histology")
    histology = str(histology).strip() if histology not in {None, ""} else None
    primary = _nullable_int(obj.get("highest_gleason_primary"), 1, 5)
    secondary = _nullable_int(obj.get("highest_gleason_secondary"), 1, 5)
    total = _nullable_int(obj.get("highest_gleason_total"), 2, 10)
    if primary is not None and secondary is not None:
        calculated_total = primary + secondary
        if total is None or total != calculated_total:
            total = calculated_total

    positive_cores = _nullable_int(obj.get("positive_cores"), 0, 200)
    total_cores = _nullable_int(obj.get("total_cores"), 0, 200)
    if positive_cores is not None and total_cores is not None and positive_cores > total_cores:
        positive_cores = None
        total_cores = None

    involvement = _nullable_float(obj.get("max_tumor_involvement_percent"), 0, 100)

    def tristate(key: str) -> str:
        value = str(obj.get(key, "not_reported")).strip().lower()
        return value if value in TRISTATE else "not_reported"

    if outcome in {"negative", "atypical", "unavailable"}:
        primary = secondary = total = None
    if outcome == "unavailable":
        histology = None
        positive_cores = total_cores = None
        involvement = None

    result = {
        "biopsy_outcome": outcome,
        "histology": histology,
        "highest_gleason_primary": primary,
        "highest_gleason_secondary": secondary,
        "highest_gleason_total": total,
        "isup_grade_group": derive_isup_grade_group(primary, secondary, total),
        "positive_cores": positive_cores,
        "total_cores": total_cores,
        "max_tumor_involvement_percent": involvement,
        "perineural_invasion": tristate("perineural_invasion"),
        "lymphovascular_invasion": tristate("lymphovascular_invasion"),
        "evidence": str(obj.get("evidence") or "").strip(),
        "extraction_notes": str(obj.get("extraction_notes") or "").strip(),
    }
    return result


def parse_model_json(raw: str) -> dict[str, Any]:
    if not isinstance(raw, str) or not raw.strip():
        return {}
    candidate = extract_first_json_object(raw.strip())
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        return {}
    return normalize_extraction(obj) if isinstance(obj, dict) else {}


def regex_extract(report_text: str) -> dict[str, Any]:
    """Conservative offline baseline used for development and smoke tests."""
    text = report_text.strip()
    upper = text.upper()
    if not text or re.search(r"NO (?:BIOPSY|PATHOLOGY|TISSUE) REPORT AVAILABLE", upper):
        outcome = "unavailable"
    elif "ADENOCARCINOMA" in upper or re.search(r"GLEASON\s+(?:SCORE\s*)?[1-5]\s*\+\s*[1-5]", upper):
        outcome = "positive"
    elif re.search(r"ATYPICAL|ASAP|HIGH.GRADE\s+PIN", upper):
        outcome = "atypical"
    elif re.search(r"NO (?:TUMOR|MALIGNANCY|CARCINOMA) IDENTIFIED", upper):
        outcome = "negative"
    else:
        outcome = "indeterminate"

    scores = [(int(a), int(b)) for a, b in re.findall(
        r"GLEASON(?:\s+SCORE)?\s*\(?\s*([1-5])\s*\+\s*([1-5])\s*\)?(?:\s*=\s*\d+)?", upper
    )]
    primary = secondary = total = None
    if scores:
        primary, secondary = max(scores, key=lambda pair: (sum(pair), pair[0]))
        total = primary + secondary

    percentages = [float(v) for v in re.findall(r"(?:APPROXIMATELY\s+|ABOUT\s+)?(\d{1,3}(?:\.\d+)?)\s*%", upper)]
    invasion = "not_reported"
    if re.search(r"(?:NO|NOT)\s+(?:EVIDENCE\s+OF\s+)?PERINEURAL INVASION", upper):
        invasion = "absent"
    elif "PERINEURAL INVASION" in upper:
        invasion = "present"

    evidence_match = re.search(
        r"[^.]{0,120}(?:ADENOCARCINOMA|NO TUMOR IDENTIFIED|ATYPICAL|GLEASON)[^.]{0,180}",
        text,
        flags=re.IGNORECASE,
    )
    return normalize_extraction({
        "biopsy_outcome": outcome,
        "histology": "prostatic adenocarcinoma" if outcome == "positive" and "ADENOCARCINOMA" in upper else None,
        "highest_gleason_primary": primary,
        "highest_gleason_secondary": secondary,
        "highest_gleason_total": total,
        "max_tumor_involvement_percent": max(percentages) if percentages else None,
        "perineural_invasion": invasion,
        "lymphovascular_invasion": "not_reported",
        "evidence": evidence_match.group(0).strip() if evidence_match else "",
        "extraction_notes": "Regex baseline; review against the source report.",
    })


def query_ollama(report_text: str, model: str, retries: int, temperature: float, num_predict: int) -> str:
    try:
        import ollama
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("The ollama package is required for --backend ollama. Install requirements.txt.") from exc

    prompt = build_prompt(report_text)
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options={"temperature": temperature, "num_predict": num_predict},
            )
            return response["message"]["content"].strip()
        except Exception as exc:  # pragma: no cover - environment dependent
            last_error = exc
            if attempt < retries - 1:
                time.sleep((2 ** attempt) + random.uniform(0, 0.5))
    raise RuntimeError(f"Ollama request failed after {retries} attempts: {last_error}")


def process_dataframe(
    df: pd.DataFrame,
    report_column: str,
    extractor: Callable[[str], dict[str, Any]],
    raw_json_column: str = "extraction_raw_json",
    overwrite: bool = False,
    max_workers: int = 2,
) -> pd.DataFrame:
    if report_column not in df.columns:
        raise ValueError(f"Report column '{report_column}' not found. Columns: {list(df.columns)}")

    result = df.copy()
    if raw_json_column not in result.columns:
        result[raw_json_column] = pd.NA
    for column in EXTRACTION_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA

    def process(index: Any) -> tuple[Any, str, dict[str, Any]]:
        existing = result.at[index, raw_json_column]
        if not overwrite and isinstance(existing, str) and existing.strip():
            parsed = parse_model_json(existing)
            if parsed:
                return index, existing, parsed
        value = result.at[index, report_column]
        if pd.isna(value) or not str(value).strip():
            parsed = normalize_extraction({"biopsy_outcome": "unavailable", "evidence": ""})
        else:
            parsed = extractor(str(value).strip())
        raw = json.dumps(parsed, ensure_ascii=False)
        return index, raw, parsed

    indices = list(result.index)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for index, raw, parsed in tqdm(executor.map(process, indices), total=len(indices)):
            result.at[index, raw_json_column] = raw
            for key, value in parsed.items():
                result.at[index, key] = value
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input XLSX file")
    parser.add_argument("--output", type=Path, help="Output XLSX file")
    parser.add_argument("--sheet", default=0, help="Worksheet name or zero-based index")
    parser.add_argument("--report-column", default="Path report biopsy")
    parser.add_argument("--id-column", default="TCIA ID")
    parser.add_argument("--backend", choices=("ollama", "regex"), default="ollama")
    parser.add_argument("--model", default="llama3.2:3b")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-predict", type=int, default=700)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    if not args.input.exists():
        raise FileNotFoundError(f"Input workbook not found: {args.input}")
    sheet: str | int = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
    df = pd.read_excel(args.input, sheet_name=sheet, engine="openpyxl")
    if args.id_column not in df.columns:
        raise ValueError(f"ID column '{args.id_column}' not found. Columns: {list(df.columns)}")

    if args.backend == "regex":
        extractor = regex_extract
    else:
        def extractor(text: str) -> dict[str, Any]:
            raw = query_ollama(text, args.model, args.retries, args.temperature, args.num_predict)
            parsed = parse_model_json(raw)
            if not parsed:
                raise ValueError(f"Model returned invalid JSON: {raw[:300]}")
            return parsed

    processed = process_dataframe(
        df,
        report_column=args.report_column,
        extractor=extractor,
        overwrite=args.overwrite,
        max_workers=max(1, args.workers),
    )
    output = args.output or args.input.with_name(f"{args.input.stem}_pathology_extracted.xlsx")
    output.parent.mkdir(parents=True, exist_ok=True)
    processed.to_excel(output, index=False, engine="openpyxl")
    print(f"Processed {len(processed)} rows. Saved: {output}")
    return output


if __name__ == "__main__":
    main()
