"""No licence-restricted image may reach a publishable location.

The Piyarathne dataset is CC BY-NC-ND 4.0: no derivatives may be redistributed,
so none of its images may appear in docs/paper/figures/, the demo's sample
assets, or anything else we publish. Figures and demo samples come from MIO or
SMART-OM (CC-BY-4.0).

Run with: python -m unittest tests.test_licence_guard -v
"""

import unittest

import numpy as np

from src.utils.config import PROJECT_ROOT, load_config
from src.utils.licence_guard import (
    LicenceViolation,
    assert_clean,
    image_phash,
    publishable_files,
    restricted_hashes,
    scan,
)

CFG = load_config()


class TestPublishableTreeIsClean(unittest.TestCase):
    def test_no_restricted_image_in_a_publishable_location(self):
        violations = scan(CFG)
        self.assertEqual(violations, [], f"licence violations: {violations}")

    def test_assert_clean_passes(self):
        self.assertTrue(assert_clean(CFG))

    def test_the_guard_actually_looks_at_files(self):
        """A guard that silently finds nothing to check would pass vacuously."""
        files = publishable_files(CFG)
        self.assertGreater(len(files), 0,
                           "no images found in the publishable directories, so the "
                           "licence check would pass without checking anything")

    def test_restricted_hashes_are_loaded(self):
        hashes = restricted_hashes(CFG)
        self.assertIn("piyarathne_oral", hashes,
                      "the restricted dataset's hashes are missing, so the check "
                      "cannot detect a copy; run scripts/audit_hashes.py")
        self.assertGreater(len(hashes["piyarathne_oral"]), 0)


class TestGuardDetectsAViolation(unittest.TestCase):
    """The guard must fail when a restricted image really is placed in a
    publishable directory - otherwise the passing test above means nothing."""

    def setUp(self):
        self.planted = None
        hashes = restricted_hashes(CFG)
        if "piyarathne_oral" not in hashes:
            self.skipTest("restricted hashes unavailable")
        import json

        from src.utils.licence_guard import _hash_cache_path

        blob = json.loads(_hash_cache_path(CFG).read_text(encoding="utf-8"))
        entry = blob["piyarathne_oral"]
        source = (PROJECT_ROOT / entry["root"] / entry["entries"][0]["path"])
        if not source.exists():
            self.skipTest("restricted dataset not on disk")

        from PIL import Image

        target_dir = PROJECT_ROOT / CFG["licence"]["publishable_dirs"][0]
        target_dir.mkdir(parents=True, exist_ok=True)
        self.planted = target_dir / "_licence_guard_probe.jpg"
        with Image.open(source) as im:
            im.draft("RGB", (512, 512))
            im.convert("RGB").resize((320, 240)).save(self.planted, quality=85)

    def tearDown(self):
        if self.planted and self.planted.exists():
            self.planted.unlink()

    def test_a_resized_copy_is_detected(self):
        violations = scan(CFG)
        offenders = [v for v in violations if "_licence_guard_probe" in v["path"]]
        self.assertTrue(offenders,
                        "a resized copy of a restricted image was NOT detected")
        self.assertEqual(offenders[0]["dataset"], "piyarathne_oral")
        with self.assertRaises(LicenceViolation):
            assert_clean(CFG)


class TestHashing(unittest.TestCase):
    def test_phash_is_stable_for_the_same_image(self):
        files = publishable_files(CFG)
        if not files:
            self.skipTest("nothing to hash")
        self.assertEqual(image_phash(files[0]), image_phash(files[0]))
        self.assertIsInstance(image_phash(files[0]), np.uint64)


if __name__ == "__main__":
    unittest.main(verbosity=2)
