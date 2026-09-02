"""Lazy Julia/KNITRO session management for tvbounds_counterfactual()
(port of ``R/julia-setup.R``).

Nothing Julia-related happens at import time. The first call of
tvbounds_counterfactual() in a session starts the embedded Julia through
juliacall (whose juliapkg resolves the package's Julia dependencies declared
in ``tvbounds/juliapkg.json`` into one environment holding PythonCall and the
solver stack), sources the TVBoundsJulia module and the Python bridge, and
runs a KNITRO availability and license check. All of this is cached in the
module-level ``_state`` and runs once per Python session.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import tempfile
import warnings

from ._object import message

# Package-local session state (not user-visible, no other global side effects).
_state = {
    "julia_started": False,
    "project_ready": False,
    "module_ready": False,
    "knitro_checked": False,
    "initialized": False,
    "probe_failed": False,
    "jl": None,             # the juliacall Main module once started
    "project_mode": None,   # "juliapkg" or "activate" (the fallback)
}

_SOLVER_PACKAGES = ("KNITRO", "ForwardDiff", "Optim", "NLSolversBase",
                    "Distributions", "Parameters")


def _julia_file(*parts) -> str:
    """Path of a file shipped in ``tvbounds/julia`` of the installed package."""
    try:
        from importlib import resources
        p = resources.files("tvbounds").joinpath("julia", *parts)
        return os.fspath(p)
    except Exception:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "julia", *parts)


def _julia_main():
    """The juliacall ``Main`` module of the initialized session."""
    if not _state["initialized"]:
        _tvb_julia_setup(verbose=False)
    return _state["jl"]


def _julia_binary_discoverable() -> bool:
    """Cheap pre-check mirroring R's ``Sys.which("julia")`` test: a Julia
    binary on the PATH, one named through an environment variable, or one
    already installed by juliapkg."""
    if shutil.which("julia"):
        return True
    for var in ("JULIA_HOME", "JULIA_BINDIR", "PYTHON_JULIACALL_BINDIR",
                "PYTHON_JULIACALL_EXE", "PYTHON_JULIAPKG_EXE"):
        if os.environ.get(var, ""):
            return True
    try:
        from juliapkg.deps import STATE
        if os.path.isdir(STATE.get("install", "")):
            return True
    except Exception:
        pass
    return False


def tvb_julia_available() -> bool:
    """Is the Julia/KNITRO backend available in this session?

    Returns True when the full backend (juliacall, a Julia installation, the
    package's Julia environment, and a licensed KNITRO installation) can be
    initialized, and False otherwise, without ever raising an error. Used by
    the test suite to skip integration tests on machines without Julia or
    KNITRO; the first successful call performs the (potentially slow)
    one-time setup.
    """
    if _state["initialized"]:
        return True
    # A failed probe is cached for the session: the setup steps that can
    # fail (Julia discovery, environment resolution, the KNITRO license
    # check) all depend on state fixed before Python started, and
    # re-probing is expensive.
    if _state["probe_failed"]:
        return False
    if importlib.util.find_spec("juliacall") is None:
        return False
    if not _julia_binary_discoverable():
        return False
    try:
        # The probe must stay silent (R wraps it in suppressWarnings /
        # suppressMessages): the one-time success message is not emitted.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _tvb_julia_setup(verbose=False, _quiet=True)
        ok = True
    except Exception:
        ok = False
    if not ok:
        _state["probe_failed"] = True
    return ok


def _prepare_juliapkg():
    """Make sure juliapkg sees ``tvbounds/juliapkg.json`` and that the
    embedded Julia can find its bundled ``lld`` linker.

    juliapkg discovers dependency files as ``<sys.path entry>/<pkg>/juliapkg.json``;
    the directory holding this package is on ``sys.path`` whenever the package
    was imported normally, but a defensive append costs nothing.

    Homebrew installs ``julia`` as a symlink; the embedded runtime derives
    ``Sys.BINDIR`` from that symlink and then cannot locate the bundled
    ``lld`` next to the real binary. The linker is needed only when a package
    must be precompiled inside the session (juliapkg precompiles everything
    in a subprocess first), but appending the real ``libexec/julia``
    directory to the PATH lets Julia's fallback lookup find it.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    if os.path.isfile(os.path.join(here, "juliapkg.json")):
        parent = os.path.dirname(here)
        norm = os.path.normcase(os.path.normpath(parent))
        found = False
        for p in sys.path:
            q = os.path.normcase(os.path.normpath(os.path.abspath(p or os.getcwd())))
            if q == norm:
                found = True
                break
        if not found:
            sys.path.append(parent)
    try:
        import juliapkg
        exe = juliapkg.executable()   # resolves the environment (juliacall does this too)
    except Exception:
        return
    real_bin = os.path.dirname(os.path.realpath(exe))
    for cand in (os.path.join(real_bin, "..", "libexec", "julia"),
                 os.path.join(real_bin, "..", "libexec")):
        cand = os.path.normpath(cand)
        if os.path.isfile(os.path.join(cand, "lld")):
            path = os.environ.get("PATH", "")
            if cand not in path.split(os.pathsep):
                os.environ["PATH"] = path + os.pathsep + cand if path else cand
            break


def _tvb_julia_setup(verbose=False, _quiet=False):
    """Initialize the Julia session, the TVBoundsJulia module, and KNITRO.

    Idempotent: the full sequence runs once per Python session and is cached
    in ``_state``. On success a one-time message notes that KNITRO was found
    and that it is a commercial solver by Artelys requiring a license; on
    failure the function raises RuntimeError with installation or license
    guidance.
    """
    if _state["initialized"]:
        return True

    if importlib.util.find_spec("juliacall") is None:
        raise RuntimeError("tvbounds_counterfactual() requires the 'juliacall' "
                           "package. Install it with pip install juliacall.")

    # (1) Start the embedded Julia (juliapkg resolves the environment
    # declared in tvbounds/juliapkg.json on first use).
    if not _state["julia_started"]:
        if verbose:
            message("tvbounds: starting Julia through juliacall (the first "
                    "use resolves and precompiles the Julia dependencies, "
                    "which can take several minutes)...")
        try:
            if verbose:
                _prepare_juliapkg()
                from juliacall import Main as jl
            else:
                # juliapkg reports its environment resolution with plain
                # print() calls; keep the silent probe and verbose = False silent
                import contextlib
                import io
                with contextlib.redirect_stdout(io.StringIO()):
                    _prepare_juliapkg()
                    from juliacall import Main as jl
        except Exception as e:
            raise RuntimeError(
                "could not start Julia through juliacall: %s\nInstall Julia "
                "(>= 1.9) from https://julialang.org/downloads/ and make sure "
                "the 'julia' binary is on the PATH (juliapkg can also download "
                "it), or set the PYTHON_JULIACALL_BINDIR environment variable "
                "to its bin/ directory." % e) from e
        _state["jl"] = jl
        _state["julia_started"] = True
    jl = _state["jl"]

    # (2) The package's Julia environment. With juliapkg the solver stack
    # is already part of the active project; otherwise fall back to what
    # the R package does: copy the shipped Project.toml to a per-session
    # directory under the temporary directory (the package installation
    # directory is never written to) and activate/instantiate it there
    # (the Julia packages themselves land in the user's depot, ~/.julia).
    if not _state["project_ready"]:
        try:
            have = all(bool(jl.seval('Base.identify_package("%s") !== nothing' % p))
                       for p in _SOLVER_PACKAGES)
        except Exception:
            have = False
        if have:
            _state["project_mode"] = "juliapkg"
        else:
            shipped = _julia_file("Project.toml")
            if not os.path.exists(shipped):
                raise RuntimeError("could not locate the package's Julia files "
                                   "(tvbounds/julia); is tvbounds installed "
                                   "correctly?")
            proj = os.path.join(tempfile.gettempdir(), "tvbounds-julia-env")
            os.makedirs(proj, exist_ok=True)
            shutil.copyfile(shipped, os.path.join(proj, "Project.toml"))
            try:
                jl._tvb_project_dir = proj
                jl.seval("import Pkg;")
                if verbose:
                    jl.seval("Pkg.activate(Main._tvb_project_dir);")
                    jl.seval("Pkg.instantiate();")
                else:
                    jl.seval("Pkg.activate(Main._tvb_project_dir; io = devnull);")
                    jl.seval("Pkg.instantiate(; io = devnull);")
            except Exception as e:
                raise RuntimeError(
                    "could not activate/instantiate the Julia environment "
                    "shipped with tvbounds: %s\nInstantiating downloads the "
                    "Julia dependencies (KNITRO.jl, ForwardDiff.jl, Optim.jl, "
                    "...) on first use and requires network access." % e) from e
            _state["project_mode"] = "activate"
        _state["project_ready"] = True

    # (3) Source the TVBoundsJulia module and the Python bridge.
    if not _state["module_ready"]:
        modfile = _julia_file("tvbounds.jl")
        bridge = _julia_file("pybridge.jl")
        for f in (modfile, bridge):
            if not os.path.exists(f):
                raise RuntimeError("could not locate tvbounds/julia/%s; is "
                                   "tvbounds installed correctly?"
                                   % os.path.basename(f))
        try:
            jl._tvb_module_file = modfile
            jl.seval("Base.include(Main, Main._tvb_module_file);")
            jl._tvb_bridge_file = bridge
            jl.seval("Base.include(Main, Main._tvb_bridge_file);")
        except Exception as e:
            raise RuntimeError("could not load the TVBoundsJulia module: %s"
                               % e) from e
        _state["module_ready"] = True

    # (4) One-time KNITRO availability + license check.
    if not _state["knitro_checked"]:
        try:
            status = jl.seval("Main.TVBoundsJulia.tvb_check_knitro()")
            status = str(status)
        except Exception as e:
            status = "error: %s" % e
        if not isinstance(status, str) or not status.startswith("ok"):
            raise RuntimeError(
                "the KNITRO solver is not usable from Julia. Status: %s\n"
                "tvbounds_counterfactual() relies on the commercial KNITRO "
                "solver by Artelys. To use it: (i) install KNITRO and obtain a "
                "valid license (free academic trials at "
                "https://www.artelys.com/solvers/knitro/); (ii) make sure "
                "KNITRO.jl can locate the installation (e.g. set the KNITRODIR "
                "environment variable, with KNITRO_JL_USE_KNITRO_JLL=false, "
                "before starting Python); (iii) if the status reports return "
                "code -520, the KNITRO library that KNITRO.jl downloaded "
                "(KNITRO_jll) is newer than your license covers: pin the "
                "release your license covers from Python, e.g. "
                "`import juliapkg; juliapkg.add(\"KNITRO_jll\", "
                "\"0e6b36f8-8e90-4eb5-b54e-06f667ea875c\", version=\"=15.1.0\"); "
                "juliapkg.resolve()`; then restart Python and try again." % status)
        _state["knitro_checked"] = True
        if not _quiet:
            message("tvbounds: Julia backend initialized; KNITRO is available ("
                    + status + "). Note that KNITRO is a commercial solver "
                    "developed by Artelys and requires a valid license; this "
                    "check runs once per Python session.")

    _state["initialized"] = True
    return True
