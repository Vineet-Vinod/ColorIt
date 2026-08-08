from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from src.pipeline.progress import ProgressBar


class ProgressBarTests(unittest.TestCase):
    @patch("src.pipeline.progress.tqdm")
    def test_update_to_only_reports_new_work(self, tqdm_mock: MagicMock) -> None:
        rendered_bar = tqdm_mock.return_value
        rendered_bar.n = 3
        rendered_bar.update.side_effect = lambda amount: setattr(
            rendered_bar, "n", rendered_bar.n + amount
        )
        progress = ProgressBar("stage", total=10, unit="frame")

        progress.update_to(7)
        progress.update_to(5)

        rendered_bar.update.assert_called_once_with(4)

    @patch("src.pipeline.progress.tqdm")
    def test_successful_finish_uses_observed_total(self, tqdm_mock: MagicMock) -> None:
        rendered_bar = tqdm_mock.return_value
        rendered_bar.n = 9
        rendered_bar.total = 10
        progress = ProgressBar("stage", total=10, unit="frame")

        progress.finish()

        self.assertEqual(rendered_bar.total, 9)
        rendered_bar.refresh.assert_called_once_with()
        rendered_bar.close.assert_called_once_with()

    @patch("src.pipeline.progress.tqdm")
    def test_failure_does_not_claim_completion(self, tqdm_mock: MagicMock) -> None:
        rendered_bar = tqdm_mock.return_value
        rendered_bar.n = 4
        rendered_bar.total = 10

        with self.assertRaisesRegex(RuntimeError, "failed"):
            with ProgressBar("stage", total=10, unit="frame"):
                raise RuntimeError("failed")

        self.assertEqual(rendered_bar.total, 10)
        rendered_bar.refresh.assert_not_called()
        rendered_bar.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
