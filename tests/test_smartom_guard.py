"""The SMART-OM loader must never hand back a burned-in annotation level.

Those levels contain the same photograph with the annotation drawn into the
pixels, so training on them would teach "drawn outline = disease". The rule is
enforced in src/utils/smartom.py; these tests hold it to that.

Run with: python -m unittest tests.test_smartom_guard -v
"""

import copy
import unittest

from src.utils.config import load_config
from src.utils.smartom import (
    AnnotatedLevelError,
    count_unannotated,
    is_blocked,
    list_images,
    patient_id_of,
    smartom_root,
)

CFG = load_config()
BLOCKED_LEVELS = ("02. Region annotation", "03. Full annotation", "04. Lesion annotation")


class TestLoaderOutput(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = smartom_root(CFG)
        if not cls.root.is_dir():
            raise unittest.SkipTest("SMART-OM not present in data/raw")
        cls.images = list_images(CFG)

    def test_returns_images(self):
        self.assertGreater(len(self.images), 0)

    def test_no_annotated_level_in_any_returned_path(self):
        for path in self.images:
            text = str(path)
            for level in BLOCKED_LEVELS:
                self.assertNotIn(level, text, f"annotated level leaked: {path}")
            self.assertFalse(is_blocked(path.relative_to(self.root)), path)

    def test_every_returned_path_is_in_the_unannotated_level(self):
        level = CFG["smartom"]["unannotated_level"]
        for path in self.images:
            self.assertIn(level, path.parts, path)

    def test_count_matches_the_unannotated_files_on_disk(self):
        self.assertEqual(len(self.images), sum(count_unannotated(CFG).values()))

    def test_all_four_categories_are_reachable(self):
        """The categories are numbered 02./03./04. too, so a rule that blocked
        any path containing those prefixes would silently drop whole diseases."""
        for category in CFG["smartom"]["categories"]:
            found = list_images(CFG, category=category)
            self.assertGreater(len(found), 0, f"category returned nothing: {category}")
            for path in found:
                self.assertIn(category, path.parts)

    def test_variation_category_is_not_mistaken_for_an_annotation_level(self):
        category = "02. Variation from normal"
        self.assertIn(category, CFG["smartom"]["categories"])
        self.assertFalse(is_blocked(category))
        self.assertGreater(len(list_images(CFG, category=category)), 0)


class TestBlockedPathDetection(unittest.TestCase):
    def test_annotation_levels_are_blocked(self):
        for path in (
            "01. Normal/02. Region annotation/01. Dorsal tongue/a.jpg",
            "03. OPMD/03. Full annotation/04. Right buccal mucosa/b.jpg",
            "04. Oral Cancer/04. Lesion annotation/06. Lower lip/c.jpg",
            "03. OPMD/04. Lesion annotation/lesion json/d.json",
            "01. Normal/02. Region annotation/09. Json files/e.json",
        ):
            self.assertTrue(is_blocked(path), path)

    def test_unannotated_paths_are_allowed(self):
        for path in (
            "01. Normal/01. Unannotated/01. Dorsal tongue/a.jpg",
            "02. Variation from normal/01. Unannotated/03. Left buccal mucosa/b.jpg",
            "04. Oral Cancer/01. Unannotated/06. Lower lip/c.jpg",
        ):
            self.assertFalse(is_blocked(path), path)


class TestConfigSwitch(unittest.TestCase):
    def test_switch_is_on(self):
        self.assertTrue(CFG["smartom"]["annotated_levels_blocked"])

    def test_turning_the_switch_off_refuses_rather_than_returning_annotated_data(self):
        cfg = copy.deepcopy(CFG)
        cfg["smartom"]["annotated_levels_blocked"] = False
        with self.assertRaises(AnnotatedLevelError):
            list_images(cfg)


class TestPatientIds(unittest.TestCase):
    def test_patient_id_extraction(self):
        self.assertEqual(patient_id_of("SMITA00006_W_DT.jpg"), "SMITA00006")
        self.assertEqual(patient_id_of("x/y/smita00123_R_LB.jpeg"), "SMITA00123")
        self.assertIsNone(patient_id_of("5 - DT.jpg"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
