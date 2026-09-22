"""
Module-level test doubles.

These classes deliberately live at module level rather than inside a fixture.
A class defined *inside* a function has no importable qualified name, so
``pickle.dump`` fails with ``AttributeError: Can't pickle local object
'mock_model.<locals>.MockModel'`` — which is exactly what made the
``temp_model_file`` fixture error and take 29 tests down with it.
"""

import numpy as np


class MockModel:
    """Mock ML model that always predicts label 0 (Mono Print)."""

    def predict(self, X):
        """Return label 0 for every row."""
        return np.zeros(len(X), dtype=int)


class MockModelVaried:
    """Mock ML model whose label depends on the ``cmyk`` feature sum."""

    def predict(self, X):
        """Map each row's ink coverage to one of the five label indices."""
        predictions = []
        for _, row in X.iterrows():
            cmyk_sum = row.get("cmyk", row.sum())
            if cmyk_sum < 50:
                predictions.append(0)  # Mono
            elif cmyk_sum < 150:
                predictions.append(1)  # Color Light
            elif cmyk_sum < 250:
                predictions.append(2)  # Color Standard
            elif cmyk_sum < 300:
                predictions.append(3)  # Color Heavy
            else:
                predictions.append(4)  # Full Color
        return np.array(predictions)
