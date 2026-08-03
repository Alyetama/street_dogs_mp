#!/usr/bin/env python3
"""Fail if any file that gets committed carries a path, address or identity
that is only true of one machine.

This exists because the leaks came back twice. The rule ("never bake
env-specific drive paths into tracked scripts; use generic defaults plus a
gitignored config") held right up until someone added one more constant at the
top of a file, and nothing noticed until it was already pushed to a public
repo -- along with a private Tailscale address and two drive serial numbers.

Scope is deliberately "what git would commit", not "what is tracked today":
tools/dashboard/ is about to move from ignored to committed, and the check has
to cover it before the first commit, not after.

    python tools/detect/tests/adv_no_hardcoded_paths.py

Exit 0 clean, 1 with a file:line list. No arguments, no network, no drives.
"""
import os
import re
import subprocess
import sys

REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Files that are committed even though .gitignore's blanket '*/' hides them
# (they are added with `git add -f`). Keep in sync with what is actually
# pushed -- a file missing here is simply not checked, which is the failure
# mode this test is meant to prevent.
EXTRA = (
    'tools/dashboard/dashboard.py',
    'tools/dashboard/dashboard.config.example.json',
    'tools/dashboard/street-dogs-dashboard.service.example',
)

SKIP_EXT = ('.js', '.min.js', '.png', '.jpg', '.svg', '.ico', '.parquet',
            '.duckdb', '.engine', '.pt', '.zst', '.gz')

# Machine-generated dependency manifests. They are full of upstream build
# prefixes (/home/conda/... from conda-forge CI) and of version strings that
# look exactly like addresses -- 'nvidia-curand-cu12==10.3.9.55'. None of it
# describes THIS host, and none of it is ours to rewrite.
SKIP_FILES = re.compile(r'(?:^|/)(?:package(?:-lock)?\.json'
                        r'|.*(?:lock|freeze)\w*\.txt'
                        r'|requirements[\w.-]*\.txt)$')

# Each pattern is (name, regex, why). They are matched against source text, so
# they fire inside docstrings and comments too -- a copy-pasteable example
# containing an absolute home directory is exactly as wrong as a constant.
#
# Note what is NOT here: '/mnt/<name>'. This project's docs use /mnt/hdd and
# /mnt/jpgs as generic example paths, and unlike /home/<user> or
# /media/<user> the /mnt convention embeds no account name. Flagging it
# produced 20+ hits and zero real leaks, and a check that cries wolf gets
# switched off.
PATTERNS = (
    ('home directory',
     re.compile(r'/home/[a-z_][a-z0-9_-]*/', re.I),
     'names a user account and their layout'),
    ('removable mount',
     re.compile(r'/(?:media|Volumes)/[a-z_][a-z0-9_-]*/', re.I),
     "names this host's mount point"),
    # Anchored on address CONTEXT, not on shape. A bare dotted quad is far
    # more often a version than an address: the first cut of this check
    # reported 'autoprefixer: ^10.4.23' as a LAN address 30 times over.
    # 100.64/10 is the exception -- Tailscale's CGNAT range never collides
    # with a plausible version number, so it is matched on shape alone.
    ('tailnet address',
     re.compile(r'\b100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])'
                r'\.\d{1,3}\.\d{1,3}\b'),
     'a Tailscale CGNAT address is private network topology'),
    ('LAN address',
     re.compile(r'(?:://|\bhost[= ]|--host[= ])\s*'
                r'(?:10|192\.168|172\.(?:1[6-9]|2\d|3[01]))'
                r'\.\d{1,3}\.\d{1,3}(?:\.\d{1,3})?\b', re.I),
     'a private address used as a host is host topology'),
    ('conda env prefix',
     re.compile(r'/(?:mini|ana)(?:forge|conda)\d*/envs/'),
     'an absolute interpreter path'),
)

# Strings that must never appear: drive serial numbers, which no regex
# generalises and which identify physical hardware.
#
# Assembled from halves rather than written out. These same serials are being
# scrubbed from the repo's history with a literal search-and-replace, and a
# scanner that spells out what it hunts for gets rewritten along with the
# leak -- silently disarming itself while still reporting success.
FORBIDDEN = tuple(a + b for a, b in (('SDG4', 'JKKR'), ('65GJ', '0P5K')))

# Lines that are allowed to match: they describe the pattern rather than being
# an instance of it. Marked, not guessed -- an exemption must be visible in the
# source it exempts.
ALLOW = re.compile(r'<user>|<drive>|<path-to|<home>|<mounts>|<datasets>'
                   r'|path-check: allow')


def files():
    out = subprocess.run(['git', '-C', REPO, 'ls-files'],
                         capture_output=True, text=True, check=True).stdout
    seen = [p for p in out.splitlines() if p]
    for p in EXTRA:
        if p not in seen and os.path.exists(os.path.join(REPO, p)):
            seen.append(p)
    return [p for p in seen
            if not p.endswith(SKIP_EXT) and not SKIP_FILES.search(p)]


# Local-only files. .gitignore names them, but the repo's blanket '*/' rule
# matches first and `git add -f` overrides every rule anyway -- so a stray
# `git add -f tools/dashboard/` would sweep the real config in alongside the
# module. Only tracking catches that.
MUST_NOT_TRACK = (
    'tools/dashboard/dashboard.config.json',
    'tools/dashboard/street-dogs-dashboard.service',
    '.env',
    'proxies.txt',
)


def tracked_secrets():
    out = subprocess.run(['git', '-C', REPO, 'ls-files'] + list(MUST_NOT_TRACK),
                         capture_output=True, text=True, check=True).stdout
    return [p for p in out.splitlines() if p]


def main():
    bad = tracked_secrets()
    if bad:
        print('LOCAL-ONLY FILE IS TRACKED BY GIT:\n')
        for p in bad:
            print(f'  {p}')
        print('\n  git rm --cached <file>   (the content stays on disk)')
        return 1

    hits = []
    for rel in files():
        path = os.path.join(REPO, rel)
        try:
            with open(path, encoding='utf-8', errors='replace') as fh:
                lines = fh.read().splitlines()
        except (OSError, IsADirectoryError):
            continue
        # Never let this test flag its OWN pattern definitions.
        if os.path.abspath(path) == os.path.abspath(__file__):
            continue
        for i, ln in enumerate(lines, 1):
            if ALLOW.search(ln):
                continue
            for name, rx, why in PATTERNS:
                m = rx.search(ln)
                if m:
                    hits.append((rel, i, name, m.group(0)[:60], why))
            for lit in FORBIDDEN:
                if lit in ln:
                    hits.append((rel, i, 'hardware serial', lit,
                                 'identifies a physical disk'))

    if not hits:
        print(f'{len(files())} committed files: no machine-specific paths, '
              'addresses or serials')
        return 0
    print(f'{len(hits)} machine-specific literal(s) in files that get '
          'committed:\n')
    for rel, i, name, txt, why in hits:
        print(f'  {rel}:{i}\n      {name}: {txt}\n      -> {why}')
    print('\nMove it to config (env var or a gitignored config file), or '
          'rewrite the example with a <placeholder>.')
    return 1


if __name__ == '__main__':
    sys.exit(main())
