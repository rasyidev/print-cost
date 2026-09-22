"""
Tests that the price table has exactly one definition.

F2 of the Dokuprint agent plan: ``src/config.py`` is the only place the per-page
prices exist. Before this, the same five prices were re-hardcoded in
``tests/conftest.py``, ``main.py`` (colours), the notebook checkpoint and a
stale six-entry ``models/price_label.json`` that contradicted the model's five
classes.
"""

import re
from pathlib import Path

import pytest

from src.config import (
    CATEGORY_COLOR_MAP,
    DEFAULT_MODEL_FILE,
    MODEL_DIR,
    NUM_PRICE_CLASSES,
    PRICE_CATEGORY_MAP,
    PRICE_LABEL_MAP,
    PRICE_TABLE,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "src" / "config.py"
#: This module names the five documented prices once, to pin the business
#: contract. It is an assertion, not a second definition of the table, so the
#: repo-wide scan skips it — everything else (conftest, UI, scripts) is in scope.
SELF_PATH = Path(__file__).resolve()

PRICE_LITERALS = tuple(PRICE_LABEL_MAP.values())

SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "htmlcov",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
    "node_modules",
}
# Code and machine-readable config only: prose (README, docs/) restates the
# contract on purpose and is not a second definition of the table.
SCAN_SUFFIXES = {".py", ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini"}


class TestCanonicalTable:
    """The table itself is internally consistent."""

    def test_has_exactly_five_classes(self):
        assert NUM_PRICE_CLASSES == 5
        assert len(PRICE_TABLE) == 5
        assert sorted(PRICE_LABEL_MAP) == [0, 1, 2, 3, 4]

    def test_label_map_matches_documented_prices(self):
        assert PRICE_LABEL_MAP == {0: 500, 1: 750, 2: 1000, 3: 1500, 4: 2000}

    def test_prices_ascend_with_label(self):
        prices = [PRICE_LABEL_MAP[label] for label in sorted(PRICE_LABEL_MAP)]
        assert prices == sorted(prices)
        assert len(set(prices)) == len(prices), "duplicate prices would collapse categories"

    def test_category_map_is_the_inverse_of_the_label_map(self):
        assert set(PRICE_CATEGORY_MAP) == set(PRICE_LABEL_MAP.values())
        assert len(PRICE_CATEGORY_MAP) == len(PRICE_LABEL_MAP)

    def test_every_category_has_a_colour(self):
        assert set(CATEGORY_COLOR_MAP) == set(PRICE_CATEGORY_MAP.values())

    def test_table_rows_agree_with_derived_maps(self):
        for label, price, category, color in PRICE_TABLE:
            assert PRICE_LABEL_MAP[label] == price
            assert PRICE_CATEGORY_MAP[price] == category
            assert CATEGORY_COLOR_MAP[category] == color


class TestSingleSourceOfTruth:
    """No other file in the repo re-declares the price table."""

    def test_no_other_file_defines_the_price_table(self):
        offenders = []

        for path in REPO_ROOT.rglob("*"):
            if not path.is_file() or path.suffix not in SCAN_SUFFIXES:
                continue
            if SKIP_DIRS.intersection(path.relative_to(REPO_ROOT).parts):
                continue
            if path.resolve() in (CONFIG_PATH, SELF_PATH):
                continue

            text = path.read_text(encoding="utf-8", errors="ignore")
            found = sorted(
                literal
                for literal in PRICE_LITERALS
                if re.search(rf"(?<!\d){literal}(?!\d)", text)
            )
            # Three or more of the five prices in one file is a table, not a
            # coincidence: two could be page counts or byte sizes.
            if len(found) >= 3:
                offenders.append(
                    f"{path.relative_to(REPO_ROOT).as_posix()}: {found}"
                )

        assert offenders == [], (
            "price table is defined outside src/config.py: " + "; ".join(offenders)
        )

    def test_stale_six_entry_label_file_is_gone(self):
        """The retired kmeans-era table (6 entries for 5 classes) must not return."""
        assert not (MODEL_DIR / "price_label.json").exists()

    def test_notebook_checkpoint_is_gone(self):
        assert not (REPO_ROOT / "src" / ".ipynb_checkpoints").exists()


@pytest.mark.integration
class TestRealModelAgreesWithTheTable:
    """The pinned classifier emits exactly the labels the table maps."""

    @pytest.fixture
    def model(self):
        from src.services.model_manager import LFS_POINTER_PREFIX

        path = MODEL_DIR / DEFAULT_MODEL_FILE
        if not path.exists():
            pytest.skip(f"{path} is missing; run `git lfs pull`")
        with open(path, "rb") as f:
            if f.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX:
                pytest.skip(
                    f"{DEFAULT_MODEL_FILE} is a Git-LFS pointer; run "
                    "`git lfs pull` or `python scripts/fetch_model.py` to check "
                    "the real classifier against the price table"
                )

        from src.services.model_manager import ModelManager

        ModelManager.clear_cache()
        return ModelManager.get_default_model()

    def test_model_classes_match_the_price_table(self, model):
        classes = [int(c) for c in model.classes_]
        assert sorted(classes) == sorted(PRICE_LABEL_MAP)

    def test_model_feature_count_is_three(self, model):
        assert model.n_features_in_ == 3
