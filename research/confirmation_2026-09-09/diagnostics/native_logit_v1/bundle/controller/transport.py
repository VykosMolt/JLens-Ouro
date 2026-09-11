"""Bounded subprocess transport; process-group cleanup adapted from prefetch v3."""
from __future__ import annotations

from pathlib import Path
import os
import signal
import subprocess
import time


def stop_process(process):
    """Clean only our new-session group, including an exited leader's children."""
    def active_members():
        members = []
        for path in Path('/proc').iterdir():
            if not path.name.isdigit():
                continue
            try:
                fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
            except (FileNotFoundError, ProcessLookupError):
                continue
            if int(fields[2]) == process.pid and fields[0] not in ('Z', 'X'):
                members.append(int(path.name))
        return members

    def group_signal(signum):
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass

    if active_members():
        group_signal(signal.SIGTERM)
        deadline = time.monotonic() + 5
        while active_members() and time.monotonic() < deadline:
            process.poll()
            time.sleep(.05)
        if active_members():
            group_signal(signal.SIGKILL)
            deadline = time.monotonic() + 2
            while active_members() and time.monotonic() < deadline:
                process.poll()
                time.sleep(.05)
    process.wait(timeout=5)
    if active_members():
        raise RuntimeError('started subprocess group did not become quiescent')


def run_bounded(command, *, timeout, operation, processes=None):
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               start_new_session=True)
    record = {'operation': operation, 'pid': process.pid, 'timeout_seconds': timeout,
              'process_group_quiescent': False}
    if processes is not None:
        processes.append(record)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command, stdout, stderr)
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    except BaseException as error:
        record['error_type'] = type(error).__name__
        raise
    finally:
        try:
            stop_process(process)
            record['process_group_quiescent'] = True
        finally:
            record['returncode'] = process.returncode
            process.stdout.close()
            process.stderr.close()
