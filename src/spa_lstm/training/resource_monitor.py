"""Lightweight CPU/GPU resource usage monitor for training runs."""

from __future__ import annotations

import csv
import subprocess
import threading
import time
from pathlib import Path


def _read_cpu_percent(prev_total: int | None, prev_idle: int | None) -> tuple[float | None, int, int]:
    with Path("/proc/stat").open("r", encoding="utf-8") as f:
        line = f.readline().strip()
    parts = line.split()
    if len(parts) < 6 or parts[0] != "cpu":
        return None, prev_total or 0, prev_idle or 0

    values = [int(value) for value in parts[1:9]]
    total = sum(values)
    idle = values[3] + values[4]  # idle + iowait

    if prev_total is None or prev_idle is None:
        return None, total, idle

    delta_total = total - prev_total
    delta_idle = idle - prev_idle
    if delta_total <= 0:
        return None, total, idle

    cpu_percent = 100.0 * (1.0 - (delta_idle / delta_total))
    return max(0.0, min(100.0, cpu_percent)), total, idle


def _read_memory_percent() -> float | None:
    values: dict[str, int] = {}
    with Path("/proc/meminfo").open("r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue
            key, raw = line.split(":", 1)
            parts = raw.strip().split()
            if not parts:
                continue
            values[key] = int(parts[0])  # kB

    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    if total is None or available is None or total <= 0:
        return None
    return 100.0 * (1.0 - (available / total))


def _read_gpu_metrics() -> tuple[float | None, float | None, float | None]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except Exception:
        return None, None, None

    if proc.returncode != 0 or not proc.stdout.strip():
        return None, None, None

    first_line = proc.stdout.strip().splitlines()[0]
    parts = [part.strip() for part in first_line.split(",")]
    if len(parts) < 3:
        return None, None, None

    try:
        gpu_util = float(parts[0])
        mem_used = float(parts[1])
        mem_total = float(parts[2])
    except ValueError:
        return None, None, None

    return gpu_util, mem_used, mem_total


class ResourceMonitor:
    """Background sampler that writes resource usage CSV rows."""

    def __init__(self, output_path: Path, interval_seconds: float = 15.0) -> None:
        self.output_path = output_path
        # Never sample more frequently than every 15 seconds.
        self.interval_seconds = max(15.0, interval_seconds)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._samples = 0
        self._gpu_seen = False
        self._file = None
        self._writer = None

    def start(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.output_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(
            [
                "timestamp_unix",
                "elapsed_seconds",
                "cpu_percent",
                "ram_percent",
                "gpu_util_percent",
                "gpu_mem_used_mb",
                "gpu_mem_total_mb",
            ]
        )
        self._file.flush()
        self._thread.start()

    def stop(self) -> dict[str, float | int | bool | str]:
        self._stop_event.set()
        self._thread.join(timeout=max(2.0, self.interval_seconds * 2.0))
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
        return {
            "resource_usage_csv": str(self.output_path),
            "resource_samples": self._samples,
            "resource_interval_seconds": self.interval_seconds,
            "gpu_metrics_observed": self._gpu_seen,
        }

    def _run(self) -> None:
        start = time.time()
        prev_total: int | None = None
        prev_idle: int | None = None

        while not self._stop_event.is_set():
            now = time.time()
            elapsed = now - start

            cpu_percent, prev_total, prev_idle = _read_cpu_percent(prev_total, prev_idle)
            ram_percent = _read_memory_percent()
            gpu_util, gpu_mem_used, gpu_mem_total = _read_gpu_metrics()
            if gpu_util is not None:
                self._gpu_seen = True

            if self._writer is not None and self._file is not None:
                self._writer.writerow(
                    [
                        f"{now:.3f}",
                        f"{elapsed:.3f}",
                        "" if cpu_percent is None else f"{cpu_percent:.3f}",
                        "" if ram_percent is None else f"{ram_percent:.3f}",
                        "" if gpu_util is None else f"{gpu_util:.3f}",
                        "" if gpu_mem_used is None else f"{gpu_mem_used:.3f}",
                        "" if gpu_mem_total is None else f"{gpu_mem_total:.3f}",
                    ]
                )
                self._file.flush()
                self._samples += 1

            self._stop_event.wait(self.interval_seconds)
