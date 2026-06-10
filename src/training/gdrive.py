import base64
import threading
import time
from pathlib import Path

import requests

from src.logging_setup import get_logger

log = get_logger(__name__)


def _download(url: str, token: str, name: str, dest: Path) -> bool:
    try:
        res = requests.get(url, params={"token": token, "name": name}, timeout=300)
    except Exception as exc:  # network hiccup -> just skip
        log.warning("download %s failed: %s", name, exc)
        return False
    body = res.text.strip()
    if res.status_code != 200 or body in ("", "NOTFOUND", "forbidden"):
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(base64.b64decode(body))
    return True


def _upload(url: str, token: str, name: str, path: Path) -> bool:
    content = base64.b64encode(path.read_bytes()).decode()
    try:
        res = requests.post(url, data={"token": token, "name": name, "content": content},
                            timeout=600)
    except Exception as exc:
        log.warning("upload %s failed: %s", name, exc)
        return False
    if res.status_code != 200 or res.text.strip() != "ok":
        log.warning("upload %s failed: %s %s", name, res.status_code, res.text[:200])
        return False
    return True


def restore(run_dir, url: str, token: str) -> bool:
    run_dir = Path(run_dir)
    (run_dir / "weights").mkdir(parents=True, exist_ok=True)
    found = False
    for name in ("last.pt", "best.pt"):
        if _download(url, token, name, run_dir / "weights" / name) and name == "last.pt":
            found = True
    log.info("restore: last.pt %s", "found" if found else "not found")
    return found


def _upload_all(run_dir: Path, url: str, token: str) -> None:
    for src in (run_dir / "weights" / "last.pt",
                run_dir / "weights" / "best.pt",
                run_dir / "results.csv"):
        if src.exists():
            _upload(url, token, src.name, src)


class BackupWatcher:
    def __init__(self, run_dir, url: str, token: str, interval: int = 60):
        self.run_dir = Path(run_dir)
        self.url = url
        self.token = token
        self.interval = interval
        self._stop = threading.Event()
        last = self.run_dir / "weights" / "last.pt"
        # seed mtime so a freshly restored last.pt isn't re-uploaded right away
        self._mtime = last.stat().st_mtime if last.exists() else None
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            last = self.run_dir / "weights" / "last.pt"
            if last.exists() and last.stat().st_mtime != self._mtime:
                self._mtime = last.stat().st_mtime
                time.sleep(3)  # let torch.save finish writing before uploading
                _upload_all(self.run_dir, self.url, self.token)
                log.info("checkpoint uploaded to Google Drive")
            self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10)
        _upload_all(self.run_dir, self.url, self.token)  # final push
        log.info("final checkpoint uploaded to Google Drive")


def start_backup(run_dir, url: str, token: str, interval: int = 60) -> BackupWatcher:
    return BackupWatcher(run_dir, url, token, interval)
