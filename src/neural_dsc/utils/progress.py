"""
Run timing and progress reporting for the experiment scripts.

These scripts are long: a full production run is hours, and the point-null
baseline of the inference sensitivity study is longer still. The practical
question is almost never "did it work" but "how long will it take if I change a
parameter and run it again" -- which needs a per-phase breakdown, not just a
total, because the phases scale with different constants. Probe selection scales
with the acceptance rate, DSC construction with `max_iterations` and how permissive
the test is, plotting with nothing much at all.

So every script reports the same three things:

  * a start banner with the wall-clock time,
  * one line per phase as it completes, with its share of the run, and
  * a summary table at the end, sorted by cost.

Phases nest: a phase opened inside another is attributed to the inner one and
indented in the summary.

Usage:

    with RunTimer("hyperparameter sweep") as run:
        with run.phase("model setup"):
            ...
        for i, ctx in enumerate(contexts, 1):
            with run.phase(f"context {ctx}"):
                ...
            run.eta(i, len(contexts))
"""

import time
from contextlib import contextmanager
from datetime import datetime, timedelta


def format_duration(seconds):
    """Human-readable duration: '4.2 s', '3 min 12 s', '2 h 05 min'."""
    if seconds != seconds:  # nan
        return "?"
    if seconds < 60:
        return f"{seconds:.1f} s"
    if seconds < 3600:
        return f"{int(seconds // 60)} min {int(seconds % 60):02d} s"
    return f"{int(seconds // 3600)} h {int((seconds % 3600) // 60):02d} min"


class RunTimer:
    """
    Times a script and the phases inside it.

    Use as a context manager; the summary prints on exit, including when the
    body raises, so a crashed or interrupted run still reports where its time
    went.
    """

    def __init__(self, name, width=78):
        self.name = name
        self.width = width
        self.started = None
        # Recorded at phase *open* so the summary reads in execution order; a
        # nested phase finishes before its parent, so completion order would
        # print the inner phase above the outer one that contains it.
        self.phases = []          # [depth, label, seconds]
        self._depth = 0
        self._section = None
        self._section_started = None

    def __enter__(self):
        self.started = time.time()
        print("=" * self.width)
        print(f"{self.name.upper()}")
        print(f"started {datetime.now().isoformat(sep=' ', timespec='seconds')}")
        print("=" * self.width, flush=True)
        return self

    @contextmanager
    def phase(self, label):
        """Time one named phase; nests, and reports even if the body raises."""
        depth = self._depth
        indent = "  " * depth
        print(f"{indent}> {label} ...", flush=True)
        record = [depth, label, float("nan")]
        self.phases.append(record)
        self._depth += 1
        started = time.time()
        try:
            yield
        finally:
            self._depth = depth
            elapsed = time.time() - started
            record[2] = elapsed
            print(f"{indent}< {label}: {format_duration(elapsed)}", flush=True)

    def section(self, label):
        """
        Close the current top-level section and open a new one.

        The flat counterpart to `phase`, for scripts whose stages run in
        sequence rather than nesting. It needs no `with` block, so it can be
        dropped into an existing script as single lines without re-indenting
        the code it measures. The final section is closed by `__exit__`.
        """
        self._close_section()
        self._section = [0, label, float("nan")]
        self._section_started = time.time()
        self.phases.append(self._section)
        print(f"\n> {label} ...", flush=True)

    def _close_section(self):
        if self._section is None:
            return
        elapsed = time.time() - self._section_started
        self._section[2] = elapsed
        print(f"< {self._section[1]}: {format_duration(elapsed)}", flush=True)
        self._section = None

    def eta(self, done, total, label="items"):
        """Print progress with a projected finish time, from the rate so far."""
        if done <= 0:
            return
        elapsed = time.time() - self.started
        remaining = elapsed / done * (total - done)
        finish = datetime.now() + timedelta(seconds=remaining)
        print(f"  [{done}/{total} {label}] {format_duration(elapsed)} elapsed, "
              f"~{format_duration(remaining)} left "
              f"(≈{finish.strftime('%H:%M')})", flush=True)

    @property
    def elapsed(self):
        return time.time() - self.started

    def __exit__(self, exc_type, exc, tb):
        self._close_section()
        total = time.time() - self.started
        print("=" * self.width)
        if exc_type is KeyboardInterrupt:
            print(f"{self.name.upper()} INTERRUPTED after {format_duration(total)}")
        elif exc_type is not None:
            print(f"{self.name.upper()} FAILED after {format_duration(total)}")
        else:
            print(f"{self.name.upper()} COMPLETE in {format_duration(total)}")
        print(f"finished {datetime.now().isoformat(sep=' ', timespec='seconds')}")

        if self.phases:
            print("-" * self.width)
            print(f"  {'phase':<48}{'time':>12}{'share':>8}")
            # Only top-level phases partition the run; nested ones are shown in
            # place but their share is of the total, not of their parent, so the
            # column is read as "of the whole run" everywhere.
            for depth, label, seconds in self.phases:
                name = "  " * depth + label
                share = 100.0 * seconds / total if total > 0 and seconds == seconds else float("nan")
                share_txt = f"{share:>7.1f}%" if share == share else f"{'?':>8}"
                print(f"  {name[:48]:<48}{format_duration(seconds):>12}{share_txt}")
            accounted = sum(s for d, _, s in self.phases if d == 0 and s == s)
            print(f"  {'(unattributed)':<48}"
                  f"{format_duration(max(0.0, total - accounted)):>12}"
                  f"{100.0 * max(0.0, total - accounted) / total if total > 0 else 0:>7.1f}%")
        print("=" * self.width, flush=True)
        return False  # never swallow the exception
