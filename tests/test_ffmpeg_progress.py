from __future__ import annotations

from io import StringIO
import unittest
from unittest.mock import MagicMock

from src.pipeline.ffmpeg_utils import _consume_ffmpeg_progress, _parse_showinfo_times


class FfmpegProgressTests(unittest.TestCase):
    def test_progress_parser_uses_absolute_frame_counts(self) -> None:
        progress = MagicMock()
        output = StringIO(
            "frame=12\n"
            "fps=8.5\n"
            "progress=continue\n"
            "frame=27\n"
            "progress=end\n"
        )

        _consume_ffmpeg_progress(output, progress=progress, value_key="frame")

        self.assertEqual(progress.update_to.call_args_list[0].args, (12,))
        self.assertEqual(progress.update_to.call_args_list[1].args, (27,))

    def test_showinfo_parser_only_returns_selected_frame_times(self) -> None:
        diagnostics = StringIO(
            "[Parsed_showinfo_2] config in time_base: 1/25\n"
            "[Parsed_showinfo_2] n: 0 pts:216 pts_time:8.64 duration:1\n"
            "unrelated diagnostic\n"
            "[Parsed_showinfo_2] n: 1 pts:285 pts_time:11.4 duration:1\n"
        )

        self.assertEqual(_parse_showinfo_times(diagnostics), [8.64, 11.4])


if __name__ == "__main__":
    unittest.main()
