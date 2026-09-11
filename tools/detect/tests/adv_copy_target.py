#!/usr/bin/env python3
"""No DuckDB COPY in this repo may bind its destination as a parameter.

    python tools/detect/tests/adv_copy_target.py

WHY. DuckDB binds a COPY *target* as parameter #1, not last. So

    con.execute("copy (select ... from read_parquet(?) ...) to ?",
                [source, destination])

writes the query's output over `source` -- the file it was supposed to read.
On 2026-09-09 that destroyed two ground_animals parquets on crucial before
anyone understood why, and they had to be rebuilt from their all_data
counterparts by re-querying the API. There is no backup of the harvest.

Proven, not inferred. Passing the markers [READ1, READ2, TARGET] to

    copy (select * from read_parquet(?) union all select * from
          read_parquet(?)) to ?

fails with 'No files found that match the pattern "READ2"' -- the second read
received READ2, which means the target had already eaten READ1.

THE RULE. Inline the destination as a quoted literal and pass only reads as
parameters:

    con.execute(f"copy (...) to {_sql_path(dst)} (format parquet)", [src])

An inlined path must still be escaped -- _sql_path doubles single quotes --
because a literal is not automatically a safe literal.

WHAT THIS READS. Every .py in the repo, parsed rather than grepped, looking
only at string arguments to a `.execute(...)` call. A docstring that
describes the trap (this file, and the two tools that carry the story) is not
a violation; a live statement is. That distinction is the whole reason this
walks an AST instead of matching lines.
"""

import ast
import os
import re
import sys

ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
SKIP_DIRS = {
    '.git', '__pycache__', 'node_modules', 'archived',
    'label-studio-ml-backend'
}

# `copy ( ... ) to ?` with the target bound. The tail allows an options clause.
BOUND_TARGET = re.compile(r'\bcopy\b.*?\bto\s*\?', re.I | re.S)

_fails = []


def check(label, ok, detail=''):
    print(('ok   ' if ok else 'FAIL ') + label + ('' if ok else '  ' + detail))
    if not ok:
        _fails.append(label)


def sql_of(node):
    """The SQL text of a string or f-string argument, or None.

    For an f-string the interpolations are replaced with a placeholder, which
    is enough: a bound target is written `to ?`, never `to {x}`, and an
    interpolated target is exactly what this check wants to see.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        out = []
        for v in node.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                out.append(v.value)
            else:
                out.append('<expr>')
        return ''.join(out)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        a, b = sql_of(node.left), sql_of(node.right)
        if a is not None and b is not None:
            return a + b
    return None


def executes(tree):
    """Every (lineno, sql) passed as the first argument to a .execute call."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.attr if isinstance(
            f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else '')
        if name not in ('execute', 'executemany', 'sql', 'query'):
            continue
        if not node.args:
            continue
        sql = sql_of(node.args[0])
        if sql:
            yield node.lineno, sql


def sources():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if fn.endswith('.py'):
                yield os.path.join(dirpath, fn)


scanned = 0
statements = 0
offenders = []
unparsed = []
for path in sources():
    try:
        tree = ast.parse(open(path, encoding='utf-8', errors='replace').read())
    except SyntaxError:
        unparsed.append(os.path.relpath(path, ROOT))
        continue
    scanned += 1
    for lineno, sql in executes(tree):
        statements += 1
        if BOUND_TARGET.search(sql):
            offenders.append(f'{os.path.relpath(path, ROOT)}:{lineno}')

check(
    f'c1 every COPY destination is inlined, across {statements:,} SQL '
    f'statements in {scanned:,} python files', not offenders,
    'these bind the target as a parameter, so DuckDB will write over the '
    'first file passed: ' + ', '.join(offenders[:5]))

# A file this cannot parse is a file this cannot vouch for. Under the repo's
# own interpreter that should be none; saying so keeps the count above from
# reading as broader coverage than it has.
check('c2 every python file parsed', not unparsed,
      'not parsed: ' + ', '.join(unparsed[:5]))

# The scan is worthless if it found no COPY at all -- a refactor that renamed
# execute(), or a walk that silently covered nothing, would leave c1 passing
# on an empty set.
copies = 0
for path in sources():
    try:
        tree = ast.parse(open(path, encoding='utf-8', errors='replace').read())
    except SyntaxError:
        continue
    for _, sql in executes(tree):
        if re.search(r'\bcopy\s*\(', sql, re.I):
            copies += 1
check(f'c3 the scan actually found COPY statements to judge ({copies})', copies
      >= 5, 'found too few to trust the walk')

if _fails:
    print(f'\n{len(_fails)} copy-target check(s) FAILED: ' + '; '.join(_fails))
    sys.exit(1)
print(f'\nall clear: no COPY binds its destination '
      f'({statements:,} statements, {copies} of them COPY)')
