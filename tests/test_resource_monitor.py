from __future__ import annotations

import io
import time
from pathlib import Path
from types import SimpleNamespace

from spa_lstm.training import resource_monitor as rm


def test_read_cpu_percent_computes_delta(monkeypatch) -> None:
    calls = {"stat": 0}

    class _FakePath:
        def __init__(self, path: str) -> None:
            self.path = path

        def open(self, *_args, **_kwargs):  # noqa: ANN202
            if self.path == "/proc/stat":
                if calls["stat"] == 0:
                    calls["stat"] += 1
                    return io.StringIO("cpu  100 0 100 700 0 0 0 0\n")
                return io.StringIO("cpu  150 0 120 730 0 0 0 0\n")
            raise AssertionError(f"Unexpected path {self.path}")

    monkeypatch.setattr(rm, "Path", _FakePath)

    first, total, idle = rm._read_cpu_percent(None, None)
    second, _, _ = rm._read_cpu_percent(total, idle)

    assert first is None
    assert second is not None
    assert abs(second - 70.0) < 1e-6


def test_read_memory_percent_parses_meminfo(monkeypatch) -> None:
    class _FakePath:
        def __init__(self, path: str) -> None:
            self.path = path

        def open(self, *_args, **_kwargs):  # noqa: ANN202
            assert self.path == "/proc/meminfo"
            return io.StringIO(
                "MemTotal:        1000 kB\n"
                "MemAvailable:     250 kB\n"
            )

    monkeypatch.setattr(rm, "Path", _FakePath)
    assert rm._read_memory_percent() == 75.0


def test_read_gpu_metrics_success(monkeypatch) -> None:
    proc = SimpleNamespace(returncode=0, stdout="73, 1000, 8000\n")
    monkeypatch.setattr(rm.subprocess, "run", lambda *args, **kwargs: proc)
    assert rm._read_gpu_metrics() == (73.0, 1000.0, 8000.0)


def test_read_gpu_metrics_failure_returns_none_tuple(monkeypatch) -> None:
    monkeypatch.setattr(rm.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    assert rm._read_gpu_metrics() == (None, None, None)


def test_resource_monitor_writes_csv_and_summary(tmp_path, monkeypatch) -> None:
    out_csv = tmp_path / "resource_usage.csv"

    monkeypatch.setattr(rm, "_read_cpu_percent", lambda prev_total, prev_idle: (12.5, 10, 5))
    monkeypatch.setattr(rm, "_read_memory_percent", lambda: 40.0)
    monkeypatch.setattr(rm, "_read_gpu_metrics", lambda: (20.0, 256.0, 1024.0))

    monitor = rm.ResourceMonitor(out_csv, interval_seconds=0.01)
    monitor.start()
    time.sleep(0.05)
    summary = monitor.stop()

    assert out_csv.exists()
    lines = out_csv.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 2  # header + at least one sample
    assert lines[0].startswith("timestamp_unix,elapsed_seconds,cpu_percent,ram_percent")

    assert summary["resource_usage_csv"] == str(out_csv)
    assert summary["resource_samples"] >= 1
    assert summary["gpu_metrics_observed"] is True
    assert summary["resource_interval_seconds"] == 15.0


def test_resource_monitor_appends_and_continues_elapsed_on_resume(tmp_path, monkeypatch) -> None:
    out_csv = tmp_path / "resource_usage.csv"
    out_csv.write_text(
        "\n".join(
            [
                "timestamp_unix,elapsed_seconds,cpu_percent,ram_percent,gpu_util_percent,gpu_mem_used_mb,gpu_mem_total_mb",
                "100.000,12.500,10.000,20.000,30.000,256.000,1024.000",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(rm, "_read_cpu_percent", lambda prev_total, prev_idle: (12.5, 10, 5))
    monkeypatch.setattr(rm, "_read_memory_percent", lambda: 40.0)
    monkeypatch.setattr(rm, "_read_gpu_metrics", lambda: (20.0, 256.0, 1024.0))

    monitor = rm.ResourceMonitor(out_csv, interval_seconds=0.01, append=True)
    monitor.start()
    time.sleep(0.05)
    monitor.stop()

    lines = out_csv.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 3
    assert lines[0].startswith("timestamp_unix,elapsed_seconds,cpu_percent,ram_percent")
    assert sum(1 for line in lines if line.startswith("timestamp_unix,")) == 1

    last_elapsed = float(lines[-1].split(",")[1])
    assert last_elapsed >= 12.5
