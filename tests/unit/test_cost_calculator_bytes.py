"""
Tests for the bytes-in pricing entry point.

``calculate_cost_from_bytes`` is the single validation + pricing path shared by
the HTTP API and in-process callers (the Dokuprint agent tool). These tests
prove the bytes call and the path call produce identical numbers, and that every
validation failure raises a typed exception rather than a guessed price.
"""

import pickle
from pathlib import Path

import pytest

from src.config import DEFAULT_DPI, PRICE_CATEGORY_MAP, PRICE_LABEL_MAP
from src.exceptions import (
    FileSizeError,
    InvalidPDFError,
    PageCountError,
    PredictionError,
)
from src.services import cost_calculator as cc
from src.services.cost_calculator import CostCalculator, calculate_cost_from_bytes
from tests.fakes import MockModelVaried


def _without_processing_time(result):
    """Drop the one field that legitimately varies between runs."""
    return {k: v for k, v in result.items() if k != "processing_time"}


class TestCalculateCostFromBytes:
    """Happy-path behaviour of the bytes entry point."""

    def test_prices_a_pdf_from_bytes(self, sample_pdf_bytes, temp_model_file):
        """A 3-page fixture PDF prices from raw bytes."""
        result = calculate_cost_from_bytes(
            sample_pdf_bytes,
            dpi=DEFAULT_DPI,
            calculator=CostCalculator(model_path=temp_model_file),
        )

        assert result["total_pages"] == 3
        assert result["total_price"] == PRICE_LABEL_MAP[0] * 3
        assert result["processing_time"] >= 0
        assert len(result["details"]) == 1
        assert result["details"][0]["category"] == PRICE_CATEGORY_MAP[PRICE_LABEL_MAP[0]]

    def test_matches_path_based_call_exactly(self, sample_pdf_bytes, sample_pdf_file, temp_model_file):
        """Bytes input and path input return the same numbers."""
        from_bytes = calculate_cost_from_bytes(
            sample_pdf_bytes,
            dpi=DEFAULT_DPI,
            calculator=CostCalculator(model_path=temp_model_file),
        )
        from_path = CostCalculator(model_path=temp_model_file).calculate_cost(
            sample_pdf_file, dpi=DEFAULT_DPI
        )

        assert _without_processing_time(from_bytes) == _without_processing_time(from_path)

    def test_accepts_bytearray_and_memoryview(self, sample_pdf_bytes, temp_model_file):
        """Byte-like payloads are accepted, not just exact ``bytes``."""
        calculator = CostCalculator(model_path=temp_model_file)

        from_bytearray = calculate_cost_from_bytes(
            bytearray(sample_pdf_bytes), calculator=calculator
        )
        from_memoryview = calculate_cost_from_bytes(
            memoryview(sample_pdf_bytes), calculator=calculator
        )

        assert from_bytearray["total_price"] == from_memoryview["total_price"]

    def test_mixed_breakdown_uses_canonical_prices(self, sample_pdf_bytes, tmp_path):
        """A multi-category document prices every category from the table."""
        model_path = tmp_path / "varied_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(MockModelVaried(), f)

        result = calculate_cost_from_bytes(
            sample_pdf_bytes,
            dpi=DEFAULT_DPI,
            calculator=CostCalculator(model_path=str(model_path)),
        )

        prices = [detail["price"] for detail in result["details"]]
        assert prices, "expected at least one category"
        assert set(prices) <= set(PRICE_LABEL_MAP.values())
        assert sum(detail["subtotal"] for detail in result["details"]) == result["total_price"]
        assert sum(detail["pages"] for detail in result["details"]) == result["total_pages"]

    def test_default_calculator_is_used_when_none_passed(self, sample_pdf_bytes, temp_model_file, monkeypatch):
        """With no calculator passed the function builds one itself."""
        monkeypatch.setattr(
            cc, "CostCalculator", lambda *a, **k: CostCalculator(model_path=temp_model_file)
        )

        result = calculate_cost_from_bytes(sample_pdf_bytes)

        assert result["total_pages"] == 3


class TestCalculateCostFromBytesValidation:
    """Every rejection path raises a typed exception."""

    def test_non_bytes_payload_rejected(self):
        with pytest.raises(InvalidPDFError) as exc_info:
            calculate_cost_from_bytes("not bytes")  # type: ignore[arg-type]

        assert "bytes" in str(exc_info.value).lower()

    def test_empty_payload_rejected(self):
        with pytest.raises(InvalidPDFError):
            calculate_cost_from_bytes(b"")

    def test_payload_without_pdf_header_rejected(self):
        with pytest.raises(InvalidPDFError) as exc_info:
            calculate_cost_from_bytes(b"<html>definitely not a pdf</html>")

        assert "%PDF" in str(exc_info.value)

    def test_oversized_payload_rejected(self, monkeypatch):
        monkeypatch.setattr(cc, "MAX_FILE_SIZE_MB", 1)

        with pytest.raises(FileSizeError) as exc_info:
            calculate_cost_from_bytes(b"%PDF" + b"0" * (2 * 1024 * 1024))

        assert exc_info.value.max_size_mb == 1

    def test_page_count_out_of_range_rejected(self, sample_pdf_bytes, monkeypatch):
        monkeypatch.setattr(cc, "MIN_PAGES", 5)
        monkeypatch.setattr(cc, "MAX_PAGES", 10)

        with pytest.raises(PageCountError) as exc_info:
            calculate_cost_from_bytes(sample_pdf_bytes)

        assert exc_info.value.page_count == 3

    def test_unreadable_pdf_stream_rejected(self):
        """A %PDF header on garbage fails as an invalid PDF, not a crash."""
        with pytest.raises((InvalidPDFError, PredictionError)):
            calculate_cost_from_bytes(b"%PDF-1.7\n" + b"garbage" * 100)


@pytest.mark.filterwarnings(
    "ignore:Using `httpx` with `starlette.testclient` is deprecated:Warning"
)
class TestHttpApiSharesTheSamePath:
    """The FastAPI endpoint delegates to ``calculate_cost_from_bytes``."""

    @pytest.fixture
    def client(self, temp_model_file, monkeypatch):
        """FastAPI TestClient whose calculator uses a mock model."""
        import importlib.util

        from fastapi.testclient import TestClient

        repo_root = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location(
            "print_cost_api", repo_root / "main-fastapi.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        monkeypatch.setattr(
            module, "get_calculator", lambda: CostCalculator(model_path=temp_model_file)
        )
        # No `with` block: the lifespan preloads the real default model, which
        # is irrelevant here and would make the test depend on the LFS artifact.
        return TestClient(module.app)

    def test_endpoint_prices_uploaded_bytes(self, client, sample_pdf_bytes):
        response = client.post(
            "/api/v1/calculate-cost",
            files={"file": ("document.pdf", sample_pdf_bytes, "application/pdf")},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["total_pages"] == 3
        assert body["total_price"] == PRICE_LABEL_MAP[0] * 3

    def test_endpoint_rejects_non_pdf_payload(self, client):
        response = client.post(
            "/api/v1/calculate-cost",
            files={"file": ("document.pdf", b"not a pdf at all", "application/pdf")},
        )

        assert response.status_code == 400
        assert "Invalid or corrupted PDF" in response.json()["detail"]

    def test_endpoint_rejects_non_pdf_extension(self, client, sample_pdf_bytes):
        response = client.post(
            "/api/v1/calculate-cost",
            files={"file": ("document.txt", sample_pdf_bytes, "text/plain")},
        )

        assert response.status_code == 400
        assert "Only PDF files" in response.json()["detail"]
