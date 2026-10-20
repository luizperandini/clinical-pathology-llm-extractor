import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pathology_extractor import (
    derive_isup_grade_group,
    main,
    normalize_extraction,
    parse_model_json,
    regex_extract,
)


POSITIVE_REPORT = (
    "RIGHT PZ: PROSTATIC ADENOCARCINOMA, GLEASON SCORE 3+4=7/10, "
    "INVOLVING ONE OF TWO CORES AND 25% OF TOTAL TISSUE. "
    "LEFT PZ: PROSTATIC ADENOCARCINOMA, GLEASON SCORE 4+3=7/10, "
    "INVOLVING TWO OF TWO CORES AND 70% OF TOTAL TISSUE. "
    "PERINEURAL INVASION IS SEEN."
)


class PathologyExtractorTests(unittest.TestCase):
    def test_grade_group_derivation(self):
        self.assertEqual(derive_isup_grade_group(3, 3, 6), 1)
        self.assertEqual(derive_isup_grade_group(3, 4, 7), 2)
        self.assertEqual(derive_isup_grade_group(4, 3, 7), 3)
        self.assertEqual(derive_isup_grade_group(4, 4, 8), 4)
        self.assertEqual(derive_isup_grade_group(5, 4, 9), 5)

    def test_json_recovery_from_markdown(self):
        raw = 'Result:\n```json\n' + json.dumps({
            "biopsy_outcome": "positive",
            "histology": "adenocarcinoma",
            "highest_gleason_primary": 4,
            "highest_gleason_secondary": 3,
            "highest_gleason_total": 7,
            "perineural_invasion": "present",
            "lymphovascular_invasion": "not_reported",
            "evidence": "GLEASON SCORE 4+3=7/10",
            "extraction_notes": "",
        }) + '\n```'
        parsed = parse_model_json(raw)
        self.assertEqual(parsed["highest_gleason_total"], 7)
        self.assertEqual(parsed["isup_grade_group"], 3)

    def test_consistency_rules(self):
        parsed = normalize_extraction({
            "biopsy_outcome": "negative",
            "highest_gleason_primary": 4,
            "highest_gleason_secondary": 3,
            "highest_gleason_total": 7,
            "positive_cores": 8,
            "total_cores": 6,
        })
        self.assertIsNone(parsed["highest_gleason_total"])
        self.assertIsNone(parsed["positive_cores"])
        self.assertIsNone(parsed["total_cores"])

    def test_regex_selects_highest_score_and_involvement(self):
        parsed = regex_extract(POSITIVE_REPORT)
        self.assertEqual(parsed["biopsy_outcome"], "positive")
        self.assertEqual(parsed["highest_gleason_primary"], 4)
        self.assertEqual(parsed["highest_gleason_secondary"], 3)
        self.assertEqual(parsed["isup_grade_group"], 3)
        self.assertEqual(parsed["max_tumor_involvement_percent"], 70.0)
        self.assertEqual(parsed["perineural_invasion"], "present")

    def test_end_to_end_workbook_regex_backend(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "input.xlsx"
            output = Path(temp_dir) / "output.xlsx"
            pd.DataFrame({
                "TCIA ID": ["SYNTH-001", "SYNTH-002"],
                "Path report biopsy": [
                    POSITIVE_REPORT,
                    "RIGHT AND LEFT BIOPSIES: BENIGN PROSTATIC TISSUE. NO TUMOR IDENTIFIED.",
                ],
            }).to_excel(source, index=False)
            main(["--input", str(source), "--output", str(output), "--backend", "regex"])
            result = pd.read_excel(output)
            self.assertEqual(result.loc[0, "biopsy_outcome"], "positive")
            self.assertEqual(result.loc[1, "biopsy_outcome"], "negative")
            self.assertEqual(result.loc[0, "highest_gleason_total"], 7)


if __name__ == "__main__":
    unittest.main()
