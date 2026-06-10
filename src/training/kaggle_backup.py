import json
import shutil
import subprocess
import threading
import time
from pathlib import Path

from src.logging_setup import get_logger

log = get_logger(__name__)


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    log.info("$ %s", " ".join(cmd))
    return subprocess.run(cmd, capture_output=True, text=True)


def _pull(dataset_slug: str, weights_dir: Path) -> tuple[bool, bool]:
    weights_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path("/tmp/cow_ckpt_pull")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    res = _run(["kaggle", "datasets", "download", dataset_slug, "-p", str(tmp), "--unzip"])
    if res.returncode != 0:
        log.info("checkpoint dataset %s not found; starting fresh", dataset_slug)
        return False, False

    resumed = False
    for name in ("last.pt", "best.pt"):
        src = tmp / name
        if src.exists():
            shutil.copy(src, weights_dir / name)
            if name == "last.pt":
                resumed = True
    log.info("restored checkpoint from %s (resume=%s)", dataset_slug, resumed)
    return True, resumed


class _BackupWatcher:
    def __init__(self, dataset_slug, title, weights_dir, stage_dir, exists, interval):
        self.slug = dataset_slug
        self.weights_dir = Path(weights_dir)
        self.stage_dir = Path(stage_dir)
        self.stage_dir.mkdir(parents=True, exist_ok=True)
        (self.stage_dir / "dataset-metadata.json").write_text(
            json.dumps({"id": dataset_slug, "title": title})
        )
        self.exists = exists
        self.interval = interval
        self._stop = threading.Event()
        last = self.weights_dir / "last.pt"
        self._last_mtime = last.stat().st_mtime if last.exists() else None
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def _push(self, msg: str):
        last = self.weights_dir / "last.pt"
        if not last.exists():
            return
        for name in ("last.pt", "best.pt"):
            src = self.weights_dir / name
            if src.exists():
                shutil.copy(src, self.stage_dir / name)
        csv = self.weights_dir.parent / "results.csv"
        if csv.exists():
            shutil.copy(csv, self.stage_dir / "results.csv")

        if self.exists:
            res = _run(["kaggle", "datasets", "version", "-p", str(self.stage_dir),
                        "-m", msg, "--dir-mode", "zip"])
        else:
            res = _run(["kaggle", "datasets", "create", "-p", str(self.stage_dir),
                        "--dir-mode", "zip"])
            if res.returncode == 0:
                self.exists = True

        if res.returncode == 0:
            log.info("checkpoint pushed to %s (%s)", self.slug, msg)
        else:
            log.warning("kaggle push failed: %s", res.stderr.strip())

    def _loop(self):
        while not self._stop.is_set():
            last = self.weights_dir / "last.pt"
            if last.exists():
                mtime = last.stat().st_mtime
                if mtime != self._last_mtime:
                    self._last_mtime = mtime
                    time.sleep(3)
                    self._push("auto checkpoint")
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=10)
        self._push("final checkpoint")


def setup_kaggle_backup(
    dataset_slug: str,
    title: str,
    weights_dir,
    stage_dir: str = "/kaggle/working/ckpt_upload",
    interval: int = 60,
):
    weights_dir = Path(weights_dir)
    exists, resumed = _pull(dataset_slug, weights_dir)
    watcher = _BackupWatcher(dataset_slug, title, weights_dir, stage_dir, exists, interval).start()
    return resumed, watcher
