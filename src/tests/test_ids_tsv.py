"""Cross-check ber.ids_tsv against the OFFICIAL utils/validate_submission.py (imported, not modified).

For every malformed file below, both our reader and the official validator must report a problem;
for well-formed files both must accept.
"""

import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ber.ids_tsv import (MATCHING_HEADER, FormatError, read_id_list_tsv,  # noqa: E402
                         read_source_ids, write_id_list_tsv)

_spec = importlib.util.spec_from_file_location("official_validator", ROOT / "utils" / "validate_submission.py")
official = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(official)

SOURCE_HEADER = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"


class TsvCrossCheck(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.test_dir = self.dir / "test"
        self.test_dir.mkdir()
        (self.test_dir / "test_source1.tsv").write_text(
            SOURCE_HEADER + "S1-1\tA\tX, 1\tUS\nS1-2\tB\tY, 2\tIndia\nS1-3\tC\tZ, 3\tFrance\n", encoding="utf-8")
        (self.test_dir / "test_source2.tsv").write_text(
            SOURCE_HEADER + "S2-10\tA\tX\tUS\nS2-11\tA\tX\tUS\nS2-12\tB\tY\tIndia\n", encoding="utf-8")
        (self.test_dir / "test_source3.tsv").write_text(
            SOURCE_HEADER + "S3-20\tA\tX\tUS\nS3-21\tC\tZ\tFrance\n", encoding="utf-8")
        self.required = read_source_ids(self.test_dir / "test_source1.tsv")
        self.valid = read_source_ids(self.test_dir / "test_source2.tsv") | \
            read_source_ids(self.test_dir / "test_source3.tsv")

    def tearDown(self):
        self.tmp.cleanup()

    def official_errors(self, path, check_ids=True):
        with contextlib.redirect_stdout(io.StringIO()):
            errors, _ = official.validate(str(path), None, str(self.test_dir), check_ids=check_ids)
        return errors

    def ours(self, path):
        _, issues = read_id_list_tsv(path, MATCHING_HEADER, required=self.required,
                                     valid_ids=self.valid, strict=False)
        return issues

    def write_raw(self, text, name="m.tsv", newline="\n"):
        p = self.dir / name
        with open(p, "w", encoding="utf-8", newline=newline) as f:
            f.write(text)
        return p

    # ---------------- accepted ----------------------------------------------------------------
    def test_writer_output_accepted_by_both_and_round_trips(self):
        mapping = {"S1-1": {"S2-11", "S3-20", "S2-10"}, "S1-2": {"S2-12"}, "S1-3": set()}
        p = self.dir / "matching_results.tsv"
        write_id_list_tsv(p, mapping, MATCHING_HEADER, s1_order=["S1-1", "S1-2", "S1-3"])
        text = p.read_bytes().decode("utf-8")
        self.assertEqual(text, "source1_entity_id\tmatched_entity_ids\n"
                               "S1-1\tS2-10,S2-11,S3-20\nS1-2\tS2-12\nS1-3\t\n")   # empty row keeps TAB
        self.assertEqual(self.official_errors(p), [])
        self.assertEqual(self.ours(p), [])
        back, _ = read_id_list_tsv(p, MATCHING_HEADER)
        self.assertEqual(back, {k: frozenset(v) for k, v in mapping.items()})

    def test_crlf_and_header_case_accepted_by_both(self):
        p = self.write_raw("SOURCE1_ENTITY_ID\tMatched_Entity_IDs\nS1-1\tS2-10\nS1-2\t\nS1-3\tS3-21\n",
                           newline="\r\n")
        self.assertEqual(self.official_errors(p), [])
        self.assertEqual(self.ours(p), [])
        back, _ = read_id_list_tsv(p, MATCHING_HEADER)
        self.assertEqual(back["S1-1"], frozenset({"S2-10"}))    # no stray '\r'

    def test_row_order_irrelevant(self):
        p = self.write_raw("source1_entity_id\tmatched_entity_ids\nS1-3\t\nS1-1\tS3-20,S2-10\nS1-2\tS2-12\n")
        self.assertEqual(self.official_errors(p), [])
        self.assertEqual(self.ours(p), [])

    # ---------------- rejected by both ------------------------------------------------------------
    BAD = {
        "duplicate_s1_row": "S1-1\tS2-10\nS1-1\tS2-11\nS1-2\t\nS1-3\t\n",
        "duplicate_id_in_list": "S1-1\tS2-10,S2-10\nS1-2\t\nS1-3\t\n",
        "self_match": "S1-1\tS1-2\nS1-2\t\nS1-3\t\n",
        "wrong_prefix": "S1-1\tX-10\nS1-2\t\nS1-3\t\n",
        "space_after_comma": "S1-1\tS2-10, S2-11\nS1-2\t\nS1-3\t\n",
        "missing_s1": "S1-1\tS2-10\nS1-2\t\n",
        "extra_s1": "S1-1\t\nS1-2\t\nS1-3\t\nS1-4\t\n",
        "empty_row_without_tab": "S1-1\tS2-10\nS1-2\nS1-3\t\n",
        "nonexistent_id": "S1-1\tS2-999\nS1-2\t\nS1-3\t\n",
    }

    def test_rejections_agree_with_official_validator(self):
        for name, body in self.BAD.items():
            with self.subTest(case=name):
                p = self.write_raw("source1_entity_id\tmatched_entity_ids\n" + body, name=f"{name}.tsv")
                self.assertTrue(self.official_errors(p), f"official validator accepted {name}")
                self.assertTrue(self.ours(p), f"ber.ids_tsv accepted {name}")
                with self.assertRaises(FormatError):
                    read_id_list_tsv(p, MATCHING_HEADER, required=self.required, valid_ids=self.valid)

    def test_comma_separated_file_rejected_by_both(self):
        p = self.write_raw("source1_entity_id,matched_entity_ids\nS1-1,S2-10\n")
        self.assertTrue(self.official_errors(p))
        self.assertTrue(self.ours(p))

    def test_wrong_header_rejected_by_both(self):
        p = self.write_raw("source1_entity_id\tcandidate_entity_ids\nS1-1\t\nS1-2\t\nS1-3\t\n")
        self.assertTrue(self.official_errors(p))
        self.assertTrue(self.ours(p))

    def test_writer_refuses_invalid_ids(self):
        for bad in ({"S1-1": ["S2-1", "S2-1"]}, {"S1-1": ["S2-1 "]}, {"S1-1": ["S2-1,S2-2"]}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                write_id_list_tsv(self.dir / "x.tsv", bad, MATCHING_HEADER)


if __name__ == "__main__":
    unittest.main()
