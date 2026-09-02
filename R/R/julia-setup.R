# Lazy Julia/KNITRO session management for tvbounds_counterfactual().
#
# Nothing Julia-related happens at package load time (CRAN machines
# have no Julia). The first call of tvbounds_counterfactual() in a
# session initializes the embedded Julia through JuliaCall, activates
# and instantiates the package's Julia environment (inst/julia),
# sources the TVBoundsJulia module, and runs a KNITRO availability and
# license check. All of this is cached in the package environment
# .tvb_state and runs once per R session.

# Package-local session state (not user-visible, no global side effects).
.tvb_state <- new.env(parent = emptyenv())

#' Is the Julia/KNITRO backend available in this session?
#'
#' Returns `TRUE` when the full backend (JuliaCall, a Julia
#' installation, the package's Julia environment, and a licensed KNITRO
#' installation) can be initialized, and `FALSE` otherwise, without ever
#' raising an error. Used by the test suite to skip integration tests on
#' machines without Julia or KNITRO; the first successful call performs
#' the (potentially slow) one-time setup.
#'
#' @return A length-one logical.
#' @keywords internal
#' @noRd
tvb_julia_available <- function() {
  if (isTRUE(.tvb_state$initialized)) return(TRUE)
  # A failed probe is cached for the session: the setup steps that can
  # fail (Julia discovery, environment instantiation, the KNITRO license
  # check) all depend on state fixed before R started, and re-probing is
  # expensive (a full Julia startup per call).
  if (isTRUE(.tvb_state$probe_failed)) return(FALSE)
  if (!requireNamespace("JuliaCall", quietly = TRUE)) return(FALSE)
  # Cheap pre-check before attempting the expensive setup: a Julia
  # binary must be discoverable.
  if (Sys.which("julia") == "" &&
      !nzchar(Sys.getenv("JULIA_HOME")) &&
      !nzchar(Sys.getenv("JULIA_BINDIR"))) {
    return(FALSE)
  }
  ok <- tryCatch({
    # The probe must stay silent: JuliaCall's own bootstrap emits
    # system2() warnings when the user's Julia environment is broken.
    suppressWarnings(suppressMessages(.tvb_julia_setup(verbose = FALSE)))
    TRUE
  }, error = function(e) FALSE)
  if (!ok) .tvb_state$probe_failed <- TRUE
  ok
}

#' Initialize the Julia session, the TVBoundsJulia module, and KNITRO
#'
#' Idempotent: the full sequence runs once per R session and is cached
#' in `.tvb_state`. On success a one-time [message()] notes that KNITRO
#' was found and that it is a commercial solver by Artelys requiring a
#' license; on failure the function stops with installation or license
#' guidance.
#'
#' @param verbose Logical; passed to the Julia setup steps.
#' @return `invisible(TRUE)` on success.
#' @keywords internal
#' @noRd
.tvb_julia_setup <- function(verbose = FALSE) {
  if (isTRUE(.tvb_state$initialized)) return(invisible(TRUE))

  if (!requireNamespace("JuliaCall", quietly = TRUE)) {
    stop("tvbounds_counterfactual() requires the 'JuliaCall' package. ",
         "Install it with install.packages(\"JuliaCall\").", call. = FALSE)
  }

  # (1) Start the embedded Julia.
  if (!isTRUE(.tvb_state$julia_started)) {
    ok <- tryCatch({
      JuliaCall::julia_setup(verbose = verbose)
      TRUE
    }, error = function(e) {
      stop("could not start Julia through JuliaCall: ",
           conditionMessage(e),
           "\nInstall Julia (>= 1.9) from https://julialang.org/downloads/ ",
           "and make sure the 'julia' binary is on the PATH, or set the ",
           "JULIA_HOME environment variable to its bin/ directory.",
           call. = FALSE)
    })
    .tvb_state$julia_started <- ok
  }

  # (2) Activate and instantiate the package's Julia environment.
  # Instantiating writes a Manifest.toml next to the Project.toml, so
  # the shipped project file is copied to a per-session directory under
  # tempdir() first: the package installation directory is never
  # written to (the Julia packages themselves land in the user's
  # standard Julia depot, ~/.julia, as with any Julia environment).
  if (!isTRUE(.tvb_state$project_ready)) {
    shipped <- system.file("julia", "Project.toml", package = "tvbounds")
    if (!nzchar(shipped)) {
      stop("could not locate the package's Julia files ",
           "(inst/julia); is tvbounds installed correctly?", call. = FALSE)
    }
    proj <- file.path(tempdir(), "tvbounds-julia-env")
    if (!dir.exists(proj)) {
      dir.create(proj, recursive = TRUE)
    }
    file.copy(shipped, file.path(proj, "Project.toml"), overwrite = TRUE)
    tryCatch({
      JuliaCall::julia_assign("_tvb_project_dir", proj)
      JuliaCall::julia_command("import Pkg;")
      if (verbose) {
        JuliaCall::julia_command("Pkg.activate(_tvb_project_dir);")
        JuliaCall::julia_command("Pkg.instantiate();")
      } else {
        JuliaCall::julia_command(
          "Pkg.activate(_tvb_project_dir; io = devnull);")
        JuliaCall::julia_command("Pkg.instantiate(; io = devnull);")
      }
    }, error = function(e) {
      stop("could not activate/instantiate the Julia environment shipped ",
           "with tvbounds: ", conditionMessage(e),
           "\nInstantiating downloads the Julia dependencies (KNITRO.jl, ",
           "ForwardDiff.jl, Optim.jl, ...) on first use and requires ",
           "network access.", call. = FALSE)
    })
    .tvb_state$project_ready <- TRUE
  }

  # (3) Source the TVBoundsJulia module.
  if (!isTRUE(.tvb_state$module_ready)) {
    modfile <- system.file("julia", "tvbounds.jl", package = "tvbounds")
    if (!nzchar(modfile)) {
      stop("could not locate inst/julia/tvbounds.jl; ",
           "is tvbounds installed correctly?", call. = FALSE)
    }
    tryCatch({
      JuliaCall::julia_assign("_tvb_module_file", modfile)
      JuliaCall::julia_command("Base.include(Main, _tvb_module_file);")
    }, error = function(e) {
      stop("could not load the TVBoundsJulia module: ",
           conditionMessage(e), call. = FALSE)
    })
    .tvb_state$module_ready <- TRUE
  }

  # (4) One-time KNITRO availability + license check.
  if (!isTRUE(.tvb_state$knitro_checked)) {
    status <- tryCatch(
      JuliaCall::julia_eval("Main.TVBoundsJulia.tvb_check_knitro()"),
      error = function(e) paste0("error: ", conditionMessage(e))
    )
    if (!is.character(status) || length(status) != 1L ||
        !startsWith(status, "ok")) {
      stop("the KNITRO solver is not usable from Julia. Status: ",
           as.character(status)[1L],
           "\ntvbounds_counterfactual() relies on the commercial KNITRO ",
           "solver by Artelys. To use it: (i) install KNITRO and obtain a ",
           "valid license (free academic trials at ",
           "https://www.artelys.com/solvers/knitro/); (ii) make sure ",
           "KNITRO.jl can locate the installation (e.g. set the KNITRO_DIR ",
           "environment variable before starting R); then restart R and ",
           "try again.", call. = FALSE)
    }
    .tvb_state$knitro_checked <- TRUE
    message("tvbounds: Julia backend initialized; KNITRO is available (",
            status, "). Note that KNITRO is a commercial solver ",
            "developed by Artelys and requires a valid license; this ",
            "check runs once per R session.")
  }

  .tvb_state$initialized <- TRUE
  invisible(TRUE)
}
