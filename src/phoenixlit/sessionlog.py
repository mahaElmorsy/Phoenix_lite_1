"""phoenixlit.sessionlog
responsible of live plotting by keeping in-memory window.
When the app exits, it saves: events.jsonl + signals_occurrence.png as outputs
"""

from dataclasses import dataclass
from pathlib import Path
from collections import Counter
import json
import time

import matplotlib.pyplot as plt


@dataclass
class LoggedEvent:
    time_stamp: float
    signal: str
    confidence: float


class SessionLogger:
    def __init__(self, out_dir: Path, *, keep_seconds: float = 10.0, live_plot: bool = True):
        # Specify the folder path where results will be saved, and if it doesn't exist, create it.
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # time determined to keep the data in memory
        self.keep_seconds = float(keep_seconds)
        self.events = []
        self._start = time.time()
        self._live_plot = bool(live_plot)
        self._fig = None
        self._axis = None
        if self._live_plot:
            plt.ion()
            self._fig, self._axis = plt.subplots()
            self._setup_axes()
            self._fig.show()

    def _setup_axes(self):
        if not self._axis:
            return
        self._axis.set_title("Phoenix Lit 0.1 — Signal occurrence")
        self._axis.set_xlabel("Signal detected")
        self._axis.set_ylabel("Occurrence")

    def _remove_old_events(self):
        # clean memory
        if self.keep_seconds <= 0:
            return
        cutoff = time.time() - self.keep_seconds
        self.events = [e for e in self.events if e.time_stamp >= cutoff]

    def log(self, signal: str, confidence: float) -> None:
        self.events.append(LoggedEvent(time_stamp=time.time(), signal=str(signal), confidence=float(confidence)))
        self._remove_old_events()
        if self._live_plot:
            self._update_plot()

    def _counts(self):
        signal_counts = Counter(e.signal for e in self.events if e.signal and e.signal != "NONE")
        return dict(sorted(signal_counts.items(), key=lambda key_value: (-key_value[1], key_value[0])))

    def _update_plot(self):
        if not self._axis:
            return
        counts = self._counts()
        self._axis.clear()
        self._setup_axes()
        if not counts:
            self._axis.text(0.5, 0.5, "No signals yet", ha="center", va="center", transform=self._axis.transAxes)
        else:
            labels = list(counts.keys())
            y = [counts[k] for k in labels]
            x = list(range(len(labels)))
            self._axis.plot(x, y, marker="o")
            self._axis.set_xticks(x, labels, rotation=30, ha="right")

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def save(self) -> None:
        # events.jsonl
        jsonl = self.out_dir / "events.jsonl"
        with jsonl.open("w", encoding="utf-8") as f:
            for e in self.events:
                rec = {
                    "time_stamp": e.time_stamp,
                    "t": e.time_stamp - self._start,
                    "signal": e.signal,
                    "confidence": e.confidence,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # signals_occurrence.png
        counts = self._counts()
        figure, axis = plt.subplots()
        axis.set_title("Phoenix Lit 0.1 — Signal occurrence")
        axis.set_xlabel("Signal detected")
        axis.set_ylabel("Occurrence")

        if not counts:
            axis.text(0.5, 0.5, "No signals", ha="center", va="center", transform=axis.transAxes)
        else:
            labels = list(counts.keys())
            y = [counts[k] for k in labels]
            x = list(range(len(labels)))
            axis.plot(x, y, marker="o")
            axis.set_xticks(x, labels, rotation=30, ha="right")

        out_png = self.out_dir / "signals_occurrence.png"
        figure.tight_layout()
        figure.savefig(out_png, dpi=150)
        plt.close(figure)
