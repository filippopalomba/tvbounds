# cran-comments

## Submission

This is a new submission: the first release of tvbounds (version 0.1.0).

## Test environments

* local: macOS (Apple Silicon, arm64, Darwin 25.5), R 4.5.2

## R CMD check results

Run with `R CMD check --as-cran` on the built source tarball:

0 errors | 0 warnings | 1 note

* The only NOTE is the standard "checking CRAN incoming feasibility ...
  New submission" NOTE. (Locally a second NOTE reports that README.md and
  NEWS.md cannot be checked because the checking machine lacks pandoc;
  this is an artifact of the local toolchain, not of the package.)

## SystemRequirements: Julia and KNITRO

The DESCRIPTION declares `SystemRequirements: For tvbounds_counterfactual():
Julia (>= 1.9) and the Artelys KNITRO solver with a valid license.` Some
context on why this does not affect checking:

* Julia and KNITRO are runtime requirements of exactly one exported
  function, `tvbounds_counterfactual()`. Nothing Julia-related runs at
  install time, load time, or check time: the backend initializes lazily on
  the first call of that function in a session.
* All Julia usage goes through the 'JuliaCall' package, which is in
  Suggests, and every use is guarded by
  `requireNamespace("JuliaCall", quietly = TRUE)` with an informative
  error when it is missing.
* The examples of `tvbounds_counterfactual()` are wrapped in `\dontrun{}`
  (they require the commercial KNITRO solver and a license, so they cannot
  be executed on check machines); the comment in the example says why. All
  tests that touch Julia or KNITRO call `skip_on_cran()` and additionally
  skip whenever Julia or KNITRO is not available on the machine.
* Every other exported function is pure R; the package and its test suite
  install, load, check, and test cleanly on machines without Julia or
  KNITRO.
* The package writes no files outside `tempdir()`: the Julia project file
  shipped in `inst/julia` is copied to a per-session directory under
  `tempdir()` before instantiation, so the installed package directory is
  never written to.

## Downstream dependencies

There are no downstream dependencies (new submission).
