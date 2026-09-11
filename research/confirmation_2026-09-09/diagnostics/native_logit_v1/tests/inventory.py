#!/usr/bin/env python3
"""Invariant 5: hash every entry under the round root except this diagnostic, or compare two inventories.

  inventory.py write OUT.json
  inventory.py compare BEFORE.json AFTER.json [--allow-added PREFIX ...]

compare exits nonzero when an existing entry changed or disappeared, or an entry
was added outside the allowed prefixes.
"""
import argparse
import json
import os
from pathlib import Path
import sys

from common import DIAGNOSTIC, ROUND, record


def inventory():
    entries = {}
    for directory, children, names in os.walk(ROUND):
        here = Path(directory)
        children[:] = sorted(c for c in children if here / c != DIAGNOSTIC)
        for name in children:
            path = here / name
            entries[path.relative_to(ROUND).as_posix() + '/'] = 'symlink' if path.is_symlink() else 'directory'
        for name in sorted(names):
            path = here / name
            key = path.relative_to(ROUND).as_posix()
            entries[key] = 'symlink:' + os.readlink(path) if path.is_symlink() else record(path)
    return entries


def compare(before, after, allowed):
    added = sorted(set(after) - set(before))
    report = {'removed': sorted(set(before) - set(after)),
              'changed': sorted(k for k in set(before) & set(after) if before[k] != after[k]),
              'added_unexpected': [k for k in added if not k.startswith(tuple(allowed))],
              'added_allowed': [k for k in added if k.startswith(tuple(allowed))]}
    report['status'] = 'unchanged' if not (report['removed'] or report['changed'] or report['added_unexpected']) else 'changed'
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('write').add_argument('out', type=Path)
    comparison = sub.add_parser('compare')
    comparison.add_argument('before', type=Path)
    comparison.add_argument('after', type=Path)
    comparison.add_argument('--allow-added', nargs='*', default=[])
    args = parser.parse_args()
    if args.command == 'write':
        entries = inventory()
        with args.out.open('x') as handle:
            json.dump(entries, handle, indent=1, sort_keys=True)
        print(json.dumps({'entries': len(entries), 'inventory': str(args.out)}))
    else:
        report = compare(json.loads(args.before.read_text()), json.loads(args.after.read_text()), args.allow_added)
        print(json.dumps(report, indent=2))
        sys.exit(report['status'] != 'unchanged')
