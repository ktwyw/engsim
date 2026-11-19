"""Parametrized pytest runner for tests/reference_checks.py (also runnable via unittest)."""

from __future__ import annotations

import unittest
import numpy as np

from reference_checks import RESULTS

try:
    import pytest
except ModuleNotFoundError:
    pytest = None


if pytest is not None:
    @pytest.mark.parametrize("section,name,ref_label,val,exp,rtol,atol", RESULTS)
    def test_reference_check(section, name, ref_label, val, exp, rtol, atol):
        assert np.isclose(val, exp, rtol=rtol, atol=atol), (
            f"[{section}] {name} vs {ref_label}: got {val}, expected {exp}"
        )


class TestReferenceChecks(unittest.TestCase):
    def test_all_reference_checks(self) -> None:
        self.assertGreaterEqual(len(RESULTS), 20)
        for section, name, ref_label, val, exp, rtol, atol in RESULTS:
            self.assertTrue(
                np.isclose(val, exp, rtol=rtol, atol=atol),
                msg=f"[{section}] {name} vs {ref_label}: got {val}, expected {exp}",
            )


if __name__ == "__main__":
    unittest.main()
