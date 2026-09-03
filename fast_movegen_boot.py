"""
Enable the Cython move generator, loudly.

engine/fast_movegen.py monkey-patches Board.legalMoves() and must be imported
BEFORE anything pulls in engine.board, so this module exists to be the first
import in every evaluation entry point.

WHY THIS IS NOT A BARE `import engine.fast_movegen`
---------------------------------------------------
Two failure modes, and silence is wrong for both.

If the .so is missing, a bare import raises and kills the run; a bare
try/except hides a ~3x slowdown behind a shrug. Evaluation runs are long
enough that a silent fallback costs hours before anyone notices. So: warn
loudly, keep running, and let the caller record which path was taken.

More important, the Cython generator returns legal moves in a DIFFERENT ORDER
from the Python one (pawns are generated set-wise). The set is identical, so
legality and correctness are unaffected -- but argmax tie-breaks in search
resolve differently. Two runs that disagree about whether the extension loaded
are therefore not bit-comparable, and since training (main.py) imports it, a
run without it is also inconsistent with the nets being evaluated. Callers
should put status() in their output metadata so that mismatch is visible in
the results rather than inferred from a missing .so months later.

Usage, as the FIRST import of the module:

    from evaluation.fast_movegen_boot import ensure_fast_movegen
    ensure_fast_movegen()
"""

import sys

_STATE = None


def ensure_fast_movegen(quiet=False):
    """Patch Board.legalMoves() with the Cython generator. Idempotent.
    Returns True if the fast path is active."""
    global _STATE
    if _STATE is not None:
        return _STATE

    if "engine.board" in sys.modules or "board" in sys.modules:
        # Not fatal -- fast_movegen patches the class object, so an already
        # imported module still picks it up -- but it means some other import
        # got there first, which is worth knowing if the ordering ever does
        # start to matter.
        if not quiet:
            print("note: engine.board was already imported before "
                  "fast_movegen; patching anyway", file=sys.stderr)

    err = None
    try:
        import engine.fast_movegen  # noqa: F401
        _STATE = True
    except Exception as e:
        # Catch Exception, not ImportError: engine/fast_movegen.py raises
        # RuntimeError if the bitboard permutation no longer matches
        # engine/board.py, and that must not kill the run either.
        #
        # Keep THIS error. The flat-layout retry below is a fallback for a
        # different repo layout, and if it also fails its message ("No module
        # named 'fast_movegen'") describes the fallback, not the cause --
        # reporting it hides the real problem, which is usually that
        # engine/movegen*.so was never built.
        err = e
        try:
            import fast_movegen  # noqa: F401  (flat layout)
            _STATE = True
        except Exception:
            _STATE = False

    if _STATE is False and not quiet:
        print("=" * 70, file=sys.stderr)
        print("WARNING: Cython move generator NOT loaded.", file=sys.stderr)
        print(f"  {type(err).__name__}: {err}", file=sys.stderr)
        print("", file=sys.stderr)
        print("Falling back to pure-Python legalMoves(). This is roughly 3x "
              "slower end", file=sys.stderr)
        print("to end and the search is move-generation bound, so a long "
              "evaluation run", file=sys.stderr)
        print("will take substantially longer than it needs to.",
              file=sys.stderr)
        print("", file=sys.stderr)
        if isinstance(err, ImportError) and "movegen" in str(err):
            print("The .pyx is not compiled. Build it:", file=sys.stderr)
        print("  pip install cython", file=sys.stderr)
        print("  python setup_movegen.py build_ext --inplace", file=sys.stderr)
        print("  python verify_movegen.py    # must print 0 mismatches",
              file=sys.stderr)
        print("  ls engine/movegen*.so       # should exist afterwards",
              file=sys.stderr)
        print("", file=sys.stderr)
        print("If the build machine and the run machine differ, set "
              "MOVEGEN_NATIVE=0 --", file=sys.stderr)
        print("setup_movegen.py uses -march=native, which can SIGILL on "
              "another node.", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
    if _STATE and not quiet:
        print("movegen: Cython (engine.fast_movegen)")
    return _STATE


def status():
    """'cython' | 'python' | 'unknown' -- for run metadata."""
    if _STATE is None:
        return "unknown"
    return "cython" if _STATE else "python"

