"""Run the current offline regression suite."""

import bootstrap  # noqa: F401
import unittest
import test_paper_protocol

if __name__ == "__main__":
    unittest.main(module=test_paper_protocol)
