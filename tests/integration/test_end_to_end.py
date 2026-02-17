"""
End-to-end integration tests.

Tests the complete workflow from PDF input to cost calculation output.
"""

import pytest
import pymupdf

from src.services.cost_calculator import CostCalculator
from src.helper import PrintCost


class TestEndToEndWorkflow:
    """Test complete PDF processing workflow."""
    
    @pytest.mark.integration
    def test_cost_calculator_with_sample_pdf(self, sample_pdf_file, temp_model_file):
        """Test CostCalculator with a sample PDF file."""
        calculator = CostCalculator(model_path=temp_model_file)
        result = calculator.calculate_cost(sample_pdf_file, dpi=7)
        
        assert result['total_pages'] == 3
        assert result['total_price'] > 0
        assert 'processing_time' in result
        assert result['processing_time'] >= 0
    
    @pytest.mark.integration
    def test_legacy_printcost_with_sample_pdf(self, sample_pdf_file, temp_model_file):
        """Test legacy PrintCost class with sample PDF."""
        pc = PrintCost(sample_pdf_file, temp_model_file)
        result = pc.predict(dpi=7)
        
        assert result['total_pages'] == 3
        assert result['total_price'] > 0
        assert len(result['details']) > 0
    
    @pytest.mark.integration
    def test_both_methods_give_same_result(self, sample_pdf_file, temp_model_file):
        """Test that both CostCalculator and PrintCost give same results."""
        # Using CostCalculator
        calculator = CostCalculator(model_path=temp_model_file)
        result_new = calculator.calculate_cost(sample_pdf_file, dpi=7)
        
        # Using legacy PrintCost
        pc = PrintCost(sample_pdf_file, temp_model_file)
        result_legacy = pc.predict(dpi=7)
        
        # Both should calculate same page count and price
        assert result_new['total_pages'] == result_legacy['total_pages']
        assert result_new['total_price'] == result_legacy['total_price']
    
    @pytest.mark.integration  
    def test_pdf_document_object_input(self, sample_pdf_bytes, temp_model_file):
        """Test with PyMuPDF Document object instead of file path."""
        pdf_doc = pymupdf.open("pdf", sample_pdf_bytes)
        
        calculator = CostCalculator(model_path=temp_model_file)
        result = calculator.calculate_cost(pdf_doc, dpi=7)
        
        assert result['total_pages'] == 3
        pdf_doc.close()
    
    @pytest.mark.integration
    def test_processing_time_reasonable(self, sample_pdf_file, temp_model_file):
        """Test that processing time is reasonable (< 5 seconds for small PDF)."""
        calculator = CostCalculator(model_path=temp_model_file)
        result = calculator.calculate_cost(sample_pdf_file, dpi=7)
        
        # 3-page PDF should process quickly
        assert result['processing_time'] < 5.0


class TestErrorHandling:
    """Test error handling in integration scenarios."""
    
    @pytest.mark.integration
    def test_invalid_pdf_path_raises_error(self, temp_model_file):
        """Test that invalid PDF path raises appropriate error."""
        from src.exceptions import InvalidPDFError
        
        calculator = CostCalculator(model_path=temp_model_file)
        
        with pytest.raises(InvalidPDFError):
            calculator.calculate_cost("/nonexistent/file.pdf", dpi=7)
    
    @pytest.mark.integration
    def test_corrupted_pdf_raises_error(self, tmp_path, temp_model_file):
        """Test that corrupted PDF raises appropriate error.  """
        from src.exceptions import InvalidPDFError
        
        # Create a fake PDF file (not really a PDF)
        fake_pdf = tmp_path / "fake.pdf"
        fake_pdf.write_text("This is not a real PDF file")
        
        calculator = CostCalculator(model_path=temp_model_file)
        
        with pytest.raises(InvalidPDFError):
            calculator.calculate_cost(str(fake_pdf), dpi=7)
