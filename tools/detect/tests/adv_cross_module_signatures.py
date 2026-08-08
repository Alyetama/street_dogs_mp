#!/usr/bin/env python3
"""Fail if a call site passes arguments the callee does not accept.

This exists because `main` shipped broken for a day and nothing noticed.
sweep.py:481 constructed engine.Consumer(..., preview_fn=preview) while the
committed engine.py had no such parameter, so every fresh clone died with

    TypeError: Consumer.__init__() got an unexpected keyword argument 'preview_fn'

It was invisible on the machine that wrote it: the matching engine.py change
sat unpushed in the working tree, so the local run was fine and only a clone
was broken. No test caught it, because the modules that would have caught it
import torch, TensorRT, cv2 and duckdb -- none of which are installed
everywhere, and two of which want a GPU that a live sweep is already using.

So this checks the AST, never imports anything, and needs no dependencies at
all. It cannot be defeated by a module that fails to load, which is exactly
the situation the real bug lived in.

    python tools/detect/tests/adv_cross_module_signatures.py

Exit 0 clean, 1 with a file:line list.

Scope and its limits, stated honestly:

  * Only callees defined at module level IN THIS REPO are checked -- we can
    see their signatures. Third-party and stdlib calls are skipped.
  * A class with no ``__init__`` of its own is skipped entirely rather than
    assumed to take nothing. It inherits one we cannot see; guessing reported
    every ``raise InvariantError("...")`` in the repo as an error.
  * ``*args``/``**kwargs`` on either side disables the relevant check, since
    anything goes through them.

That leaves real gaps (dynamic dispatch, methods on instances, re-exported
names). It is not a type checker. It catches the one mistake that actually
shipped, cheaply, everywhere, with no false positives on the current tree.
"""
import ast
import os
import subprocess
import sys

REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def tracked_py():
    out = subprocess.run(['git', '-C', REPO, 'ls-files', '*.py'],
                         capture_output=True, text=True, check=True).stdout
    return [p for p in out.splitlines() if p]


def sig_of(fn, is_method):
    a = fn.args
    pos = [x.arg for x in (a.posonlyargs + a.args)]
    if is_method and pos and pos[0] in ('self', 'cls'):
        pos = pos[1:]
    return {
        'pos': pos,
        'required': len(pos) - len(a.defaults),
        'kwonly': [x.arg for x in a.kwonlyargs],
        'star': a.vararg is not None,
        'kwstar': a.kwarg is not None,
    }


def collect(files):
    """(defs, trees): every module-level def/class signature we can resolve."""
    defs, trees = {}, {}
    for rel in files:
        try:
            with open(os.path.join(REPO, rel), encoding='utf-8') as fh:
                tree = ast.parse(fh.read(), rel)
        except (OSError, SyntaxError) as e:
            print(f'  cannot parse {rel}: {e}')
            continue
        trees[rel] = tree
        table = defs.setdefault(os.path.splitext(os.path.basename(rel))[0], {})
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                table[node.name] = sig_of(node, False)
            elif isinstance(node, ast.ClassDef):
                init = next((f for f in node.body
                             if isinstance(f, (ast.FunctionDef,
                                               ast.AsyncFunctionDef))
                             and f.name == '__init__'), None)
                if init is not None:
                    table[node.name] = sig_of(init, True)
    return defs, trees


def resolve_imports(tree, defs):
    """(alias, imported): local names that point at modules in this repo."""
    alias, imported = {}, {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                base = n.name.split('.')[-1]
                if base in defs:
                    alias[n.asname or base] = base
        elif isinstance(node, ast.ImportFrom):
            base = (node.module or '').split('.')[-1]
            if base in defs:
                for n in node.names:
                    imported[n.asname or n.name] = base
    return alias, imported


def check(rel, tree, defs):
    selfmod = os.path.splitext(os.path.basename(rel))[0]
    alias, imported = resolve_imports(tree, defs)
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f, target = node.func, None
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
            m = alias.get(f.value.id)
            if m:
                target = (m, f.attr)
        elif isinstance(f, ast.Name):
            if f.id in imported:
                target = (imported[f.id], f.id)
            elif f.id in defs.get(selfmod, {}):
                target = (selfmod, f.id)
        sig = defs.get(target[0], {}).get(target[1]) if target else None
        if not sig:
            continue

        kws = [k.arg for k in node.keywords if k.arg is not None]
        dblstar = any(k.arg is None for k in node.keywords)
        star = any(isinstance(a, ast.Starred) for a in node.args)
        npos = sum(1 for a in node.args if not isinstance(a, ast.Starred))
        where = f'{rel}:{node.lineno}'
        callee = f'{target[0]}.{target[1]}'

        if not sig['kwstar'] and not dblstar:
            bad = [k for k in kws
                   if k not in sig['pos'] and k not in sig['kwonly']]
            if bad:
                out.append((where, callee,
                            f"passes {', '.join(bad)}= which it does not accept"
                            f" (takes {sig['pos'] + sig['kwonly'] or 'nothing'})"))
        if not sig['star'] and not star and npos > len(sig['pos']):
            out.append((where, callee, f'passes {npos} positional args, '
                                       f"takes {len(sig['pos'])}"))
        if not star and not dblstar and npos < sig['required']:
            supplied = set(kws) | set(sig['pos'][:npos])
            missing = [p for p in sig['pos'][:sig['required']]
                       if p not in supplied]
            if missing:
                out.append((where, callee,
                            f"missing required {', '.join(missing)}"))
    return out


def shadowed(trees):
    """[(module, name, lines)] defined twice at module level.

    The second definition wins silently, and in a ten-thousand-line file the
    two can be a hundred lines apart: a helper added near its caller was
    shadowed by an unrelated one further down that took different arguments,
    and nothing said so until it was called. Same-name, same-module, one
    survives -- that is never intentional.
    """
    import ast as _ast
    out = []
    for rel, tree in trees.items():
        seen = {}
        for n in tree.body:
            if not isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef,
                                  _ast.ClassDef)):
                continue
            if n.name in seen:
                out.append((rel, n.name, (seen[n.name], n.lineno)))
            seen[n.name] = n.lineno
    return out


def main():
    files = tracked_py()
    defs, trees = collect(files)
    dupes = shadowed(trees)
    if dupes:
        for rel, name, (a, b) in sorted(dupes):
            print(f'FAIL {rel}:{b} redefines {name}(), already defined at '
                  f'line {a} — the first one is dead and every call to it '
                  f'silently reaches the second')
        return 1
    problems = [p for rel, tree in trees.items() for p in check(rel, tree, defs)]
    if not problems:
        n = sum(len(v) for v in defs.values())
        print(f'{len(trees)} modules, {n} resolvable symbols: '
              'no shadowed definitions, no cross-module signature mismatches')
        return 0
    print(f'{len(problems)} call site(s) disagree with their callee:\n')
    for where, callee, why in sorted(problems):
        print(f'  {where}\n      {callee}(): {why}')
    return 1


if __name__ == '__main__':
    sys.exit(main())
