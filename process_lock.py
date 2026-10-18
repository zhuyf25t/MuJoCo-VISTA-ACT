"""Short cross-process filesystem critical sections on local disks."""

import os
import time
from pathlib import Path


class FileLock:
    """Local process lock released by the OS even after a crash.

    Keep the file in place: unlinking it can allow different locks for the same
    pathname. Intended for local Windows/Linux filesystems, not NFS.
    """
    def __init__(self, path: Path, timeout: float = 10.0):
        self.path, self.timeout = path, timeout

    def __enter__(self):
        self.file = self.path.open("a+b")
        if self.file.seek(0, os.SEEK_END) == 0:
            self.file.write(b"\0")
            self.file.flush()
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                self.file.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except OSError as error:
                if time.monotonic() >= deadline:
                    self.file.close()
                    raise BlockingIOError(f"记录目录正在使用: {self.path}") from error
                time.sleep(0.05)

    def __exit__(self, *exc):
        self.file.close()
