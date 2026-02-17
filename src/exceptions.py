"""
Custom exception classes for the Print Cost application.

This module defines specific exception types to enable better error handling
and make error cases more explicit throughout the codebase.
"""


class PrintCostBaseException(Exception):
    """Base exception class for all Print Cost application exceptions."""
    pass


class InvalidFileTypeError(PrintCostBaseException):
    """Raised when a file with an unsupported type is provided."""
    
    def __init__(self, file_type: str, supported_types: list[str]):
        self.file_type = file_type
        self.supported_types = supported_types
        super().__init__(
            f"Invalid file type: '{file_type}'. "
            f"Supported types: {', '.join(supported_types)}"
        )


class InvalidPDFError(PrintCostBaseException):
    """Raised when a PDF file is corrupted or cannot be processed."""
    
    def __init__(self, message: str = "Invalid or corrupted PDF file"):
        super().__init__(message)


class ModelLoadError(PrintCostBaseException):
    """Raised when a machine learning model fails to load."""
    
    def __init__(self, model_path: str, original_error: Exception = None):
        self.model_path = model_path
        self.original_error = original_error
        message = f"Failed to load model from: {model_path}"
        if original_error:
            message += f"\nOriginal error: {str(original_error)}"
        super().__init__(message)


class PredictionError(PrintCostBaseException):
    """Raised when model prediction fails."""
    
    def __init__(self, message: str = "Prediction failed", original_error: Exception = None):
        self.original_error = original_error
        if original_error:
            message += f"\nOriginal error: {str(original_error)}"
        super().__init__(message)


class ValidationError(PrintCostBaseException):
    """Raised when input validation fails."""
    
    def __init__(self, field: str, message: str):
        self.field = field
        super().__init__(f"Validation failed for '{field}': {message}")


class FileSizeError(PrintCostBaseException):
    """Raised when a file exceeds the maximum allowed size."""
    
    def __init__(self, file_size_mb: float, max_size_mb: int):
        self.file_size_mb = file_size_mb
        self.max_size_mb = max_size_mb
        super().__init__(
            f"File size ({file_size_mb:.2f} MB) exceeds maximum allowed size ({max_size_mb} MB)"
        )


class PageCountError(PrintCostBaseException):
    """Raised when PDF page count is outside acceptable range."""
    
    def __init__(self, page_count: int, min_pages: int, max_pages: int):
        self.page_count = page_count
        self.min_pages = min_pages
        self.max_pages = max_pages
        super().__init__(
            f"Invalid page count: {page_count}. Must be between {min_pages} and {max_pages}"
        )
