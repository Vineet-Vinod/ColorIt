from __future__ import annotations

from types import TracebackType

from tqdm.auto import tqdm


class ProgressBar:
    """A small tqdm wrapper that keeps pipeline progress reporting consistent."""

    def __init__(
        self,
        description: str,
        *,
        total: int | float | None,
        unit: str,
        initial: int | float = 0,
    ) -> None:
        self._bar = tqdm(
            total=total,
            desc=description,
            unit=unit,
            initial=initial,
            dynamic_ncols=True,
            mininterval=0.25,
            smoothing=0.15,
            leave=True,
            position=0,
        )

    @property
    def completed(self) -> int | float:
        return self._bar.n

    def update(self, amount: int | float = 1) -> None:
        if amount > 0:
            self._bar.update(amount)

    def update_to(self, completed: int | float) -> None:
        self.update(max(0, completed - self._bar.n))

    def finish(self) -> None:
        """Close at the observed total when metadata and decoded work disagree."""
        if self._bar.total is not None and self._bar.n != self._bar.total:
            self._bar.total = self._bar.n
            self._bar.refresh()
        self._bar.close()

    def close(self) -> None:
        self._bar.close()

    def __enter__(self) -> ProgressBar:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_type is None:
            self.finish()
        else:
            self.close()
