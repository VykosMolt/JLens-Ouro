#!/usr/bin/env python3
"""Rehash the protected roots and compare them with inventory/protected_before.json (bytes, mtime, SHA-256, git state)."""
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

V = Path(__file__).resolve().parents[1]
BEFORE = json.loads((V / 'inventory/protected_before.json').read_text())


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 24), b''):
            digest.update(block)
    return digest.hexdigest()


def git(repo):
    run = lambda *args: subprocess.run(['git', '-C', repo, *args], capture_output=True, text=True).stdout  # noqa: E731
    return {'head': run('rev-parse', 'HEAD').strip(), 'status_short': run('status', '--short')}


def main():
    started = datetime.now(timezone.utc).isoformat()
    result = {'schema': 'verification_protected_comparison.v1', 'started_utc': started, 'roots': {}}
    for root, before in BEFORE['roots'].items():
        found = set()
        for directory, _, names in os.walk(root, followlinks=False):
            for name in names:
                found.add((Path(directory) / name).relative_to(root).as_posix())
        changed = []
        for rel in sorted(found & set(before)):
            path = Path(root) / rel
            stat = path.stat()
            now = {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'sha256': sha256(path)}
            if now != before[rel]:
                changed.append({'path': rel, 'before': before[rel], 'after': now})
        result['roots'][root] = {'files_before': len(before), 'files_after': len(found), 'changed': changed,
                                 'removed': sorted(set(before) - found), 'added': sorted(found - set(before))}
    result['git'] = {repo: {'before': state, 'after': git(repo)} for repo, state in BEFORE['git'].items()}
    result['git_unchanged'] = all(v['before'] == v['after'] for v in result['git'].values())
    result['unchanged'] = result['git_unchanged'] and all(not (r['changed'] or r['removed'] or r['added']) for r in result['roots'].values())
    result['finished_utc'] = datetime.now(timezone.utc).isoformat()
    with (V / 'inventory/protected_after_comparison.json').open('x') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(json.dumps({'unchanged': result['unchanged'], 'git_unchanged': result['git_unchanged'],
                      **{root: {k: len(v) if isinstance(v, list) else v for k, v in r.items()} for root, r in result['roots'].items()}}, indent=1))


if __name__ == '__main__':
    main()
