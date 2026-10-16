"""Reject tracked local environments and obvious credentials without printing values."""
from pathlib import Path
import re
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
TOKEN = re.compile(rb'(?:gsk_|sk-proj-|github_pat_|ghp_)[A-Za-z0-9_-]{20,}')


def main():
    failures = []
    paths = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    for rel in filter(None, paths):
        name, path = Path(rel).name, ROOT / rel
        if name == '.env' or (name.startswith('.env.') and not name.endswith('.example')):
            failures.append(rel + ': local environment tracked')
        if path.is_file() and TOKEN.search(path.read_bytes()):
            failures.append(rel + ': credential-like token detected (value suppressed)')
    for failure in failures: print(failure)
    if not failures: print('No tracked environments or recognized credential tokens found.')
    return bool(failures)


if __name__ == '__main__': sys.exit(main())
