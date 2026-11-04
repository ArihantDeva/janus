"""Durable Amarel transport for the encoder classifier.

The installed configuration names no job id, node or run directory. Each hook
event reuses the allocation recorded in a small state file; if srun reports it
gone, the next request starts a fresh owned allocation and retries once. That
is recovery on the next explicit request -- there is no watchdog or daemon.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import subprocess

import janus_interface as JI

DEFAULT_HOST = "amarel"
DEFAULT_REMOTE_DIR = "/scratch/ad2416/janus-muse"
DEFAULT_TIMEOUT = 240


class Unavailable(RuntimeError):
    """No classifier answer. Callers must not turn this into advice."""


class Stale(Unavailable):
    """The recorded allocation no longer accepts work."""


class Timeout(Unavailable):
    """The deadline passed before the cluster answered."""


def _clean(message):
    """One short line for diagnostics; paths and credentials are dropped."""
    return re.sub(r"\s+", " ", str(message)).strip()[:160]


class SlurmLauncher:
    """Owns allocations through sbatch/srun. `run` is replaced in tests."""

    def __init__(self, host=None, remote_dir=None, run=None, timeout=None):
        self.host = host or os.environ.get("JANUS_REMOTE_HOST", DEFAULT_HOST)
        self.remote_dir = remote_dir or os.environ.get("JANUS_REMOTE_DIR",
                                                       DEFAULT_REMOTE_DIR)
        self.timeout = int(timeout or os.environ.get("JANUS_REMOTE_TIMEOUT",
                                                     DEFAULT_TIMEOUT))
        self._run = run or self._ssh

    def _call(self, argv, timeout, stdin=None):
        """One remote command, with transport failures mapped to our contract."""
        try:
            return self._run(argv, timeout, stdin)
        except Unavailable:
            raise
        except TimeoutError:
            raise Timeout("remote deadline exceeded") from None
        except OSError as error:
            raise Unavailable(type(error).__name__) from None

    def _ssh(self, argv, timeout, stdin=None):
        try:
            done = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                 "-o", "ControlMaster=auto",
                 "-o", "ControlPath=~/.ssh/janus-%r@%h-%p",
                 "-o", "ControlPersist=600",
                 self.host, " ".join(str(part) for part in argv)],
                input=stdin, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise Timeout("remote deadline exceeded") from None
        if done.returncode != 0:
            raise self._classify(done.stderr or done.stdout or "no output")
        return done.stdout

    @staticmethod
    def _classify(detail):
        low = detail.lower()
        if "invalid job" in low or "no such job" in low or "job not found" in low:
            return Stale(_clean(detail))
        return Unavailable(_clean(detail))

    def _remote(self, *parts):
        return os.path.join(self.remote_dir, *parts)

    def start(self):
        """Submit one owned allocation; wait for its socket inside that allocation."""
        out = self._call(["sbatch", "--parsable", f"--chdir={self.remote_dir}",
                          self._remote("scripts", "remote_service.sbatch")],
                         self.timeout)
        job = (out.strip().splitlines() or [""])[-1].strip()
        if not job.isdigit():
            raise Unavailable(f"sbatch did not return a job id: {_clean(out)}")
        self._call([self._remote("scripts", "remote_wait.sh"), job], self.timeout)
        return job

    def cleanup(self, job):
        """Remove the released allocation's socket; safe to call when absent."""
        self._call(["rm", "-f",
                    os.path.join(self.remote_dir, "run", f"encoder-{job}.sock"),
                    os.path.join(self.remote_dir, "run", f"encoder-{job}.sock.ready")],
                   30)

    def alive(self, job):
        try:
            out = self._call(["squeue", "-h", "-j", str(job), "-o", "%T"], 30)
        except Unavailable:
            return False
        return bool(out.strip())

    def request(self, job, text, timeout):
        """One canonical encoder input through srun, on the allocation's node."""
        out = self._call(self._request_argv(job), timeout, text)
        return self._distribution(out)

    def describe(self, job, timeout=60):
        """Ask the allocation what it serves: checkpoint, head, calibration."""
        out = self._call([os.path.join(self.remote_dir, "scripts", "remote_describe.sh"),
                          str(job)], timeout)
        return json.loads(out)

    def _request_argv(self, job):
        return ["srun", f"--jobid={job}", "--overlap", "--ntasks=1",
                "--cpus-per-task=4", self._remote("scripts", "remote_client.sh"),
                str(job)]

    @staticmethod
    def _distribution(raw):
        try:
            envelope = json.loads(raw)
        except (TypeError, ValueError):
            raise Unavailable("remote reply was not a single JSON object") from None
        if not isinstance(envelope, dict) or "distribution" not in envelope:
            raise Unavailable("remote reply carried no distribution")
        return envelope["distribution"]


class RemoteEncoder:
    """Janus classifier whose weights exist only on the cluster."""

    name = "janus"
    uses_encoder_input = True

    def __init__(self, launcher=None, state_dir=None, timeout=None):
        self.launcher = launcher or SlurmLauncher()
        self.state_dir = state_dir or default_state_dir()
        self.timeout = int(timeout or os.environ.get("JANUS_REMOTE_TIMEOUT",
                                                     DEFAULT_TIMEOUT))
        self._job = self._recorded_job()

    def _state_file(self):
        os.makedirs(self.state_dir, exist_ok=True)
        return os.path.join(self.state_dir, "allocation")

    def _recorded_job(self):
        try:
            return open(self._state_file()).read().strip() or None
        except FileNotFoundError:
            return None

    def _remember(self, job):
        self._job = job
        with open(self._state_file(), "w") as handle:
            handle.write(job)

    def __call__(self, text):
        try:
            return self._ask(self._job, text)
        except Stale:
            return self._ask(self._restart(), text)
        except (Timeout, OSError) as error:
            raise Unavailable(type(error).__name__) from None

    def _restart(self):
        """Serialized so two concurrent hook events cannot both submit."""
        with open(os.path.join(self.state_dir, ".start.lock"), "a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            recorded = self._recorded_job()
            if recorded and self.launcher.alive(recorded):
                self._job = recorded
                return recorded
            self._clear_stale_socket(recorded)
            try:
                self._remember(self.launcher.start())
            except Unavailable:
                raise
            except Exception as error:
                raise Unavailable(type(error).__name__) from None
            return self._job

    def _clear_stale_socket(self, job):
        """A released allocation leaves its socket behind; it is not a resource."""
        if not job:
            return
        try:
            self.launcher.cleanup(job)
        except AttributeError:
            return

    def _ask(self, job, text):
        if not job:
            raise Stale("no allocation recorded")
        try:
            return describe(self.launcher.request(job, text, self.timeout))
        except Unavailable:
            raise
        except Exception as error:
            raise Unavailable(type(error).__name__) from None


def describe(dist):
    """Accept only a usable distribution over the fixed encoder classes."""
    from backends import validate

    try:
        checked = validate(dist, "janus")
    except (TypeError, ValueError) as error:
        raise Unavailable(f"invalid distribution: {type(error).__name__}") from None
    return {name: checked[name] for name in JI.CLASSES_3}


def default_state_dir():
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(base, "janus-muse-encoder")
