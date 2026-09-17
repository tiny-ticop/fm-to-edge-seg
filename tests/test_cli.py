from __future__ import annotations

import contextlib
import io
import unittest

from fm_to_edge_seg import __version__
from fm_to_edge_seg.cli import main


class CliTest(unittest.TestCase):
    def test_version_is_defined(self) -> None:
        self.assertTrue(__version__)

    def test_doctor_command(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = main(["doctor"])

        self.assertEqual(exit_code, 0)
        self.assertIn("fm-to-edge-seg", output.getvalue())
        self.assertIn("python:", output.getvalue())


if __name__ == "__main__":
    unittest.main()
