from experiments.adapter_compare.windows import frame_windows


def test_windows_cover_exact_frames_and_do_not_cross_cuts() -> None:
    cuts = [1, 13, 141, 476, 625, 715, 744, 749]
    windows = frame_windows(750, 15, cuts)
    delivered = [
        frame
        for window in windows
        for frame in range(window.begin + window.skip, window.end)
    ]
    assert delivered == list(range(750))
    assert all(
        not any(window.begin < cut < window.end for cut in cuts) for window in windows
    )
    assert all(window.end - window.begin <= 15 for window in windows)
