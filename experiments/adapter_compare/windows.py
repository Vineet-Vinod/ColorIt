from itertools import pairwise

from .adapter_compare_dataclasses import FrameWindow


def frame_windows(frames: int, window: int, cuts: list[int]) -> list[FrameWindow]:
    boundaries = [0, *sorted({cut for cut in cuts if 0 < cut < frames}), frames]
    result = []
    for start, stop in pairwise(boundaries):
        begin = start
        skip = 0
        while begin < stop:
            end = min(begin + window, stop)
            result.append(FrameWindow(begin, end, skip))
            if end == stop:
                break
            begin = end - 2
            skip = 2
    return result
