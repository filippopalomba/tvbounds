# cran-comments

## Resubmission

This is a resubmission of the first release (now version 0.1.1),
addressing the three points raised in the CRAN review of 2026-08-21:

* References in the Description: the Palomba (2026) reference now carries
  an auto-linking URL, in the requested form
  authors (year) "Title" <https:...>. The paper is an unpublished working
  paper with no DOI or ISBN, so the author's page hosting it is linked.
* T and F: the package never used `T`/`F` as logical shorthand; the flag
  came from an argument of `tvbounds_riv()` named `F`. That argument is
  renamed to `Fmat` throughout the package, documentation, tests, and
  vignette.
* Writing to the .GlobalEnv: the only writes were the customary
  save-and-restore of `.Random.seed` around user-seeded computations.
  All of it now goes through withr::local_seed() /
  withr::local_preserve_seed() ('withr' added to Imports); the package
  itself no longer assigns into the global environment anywhere.

## Test environments

* local: macOS (Apple Silicon, arm64, Darwin 25.5), R 4.5.2
* win-builder: R-devel and R-release
* R-hub: Linux (R-devel)

## R CMD check results

Run with `R CMD check --as-cran` on the built source tarball:

0 errors | 0 warnings | 1 note

* The only NOTE is the standard "checking CRAN incoming feasibility ...
  New submission" NOTE.

(On the local machine a second NOTE appears under "checking HTML version of
manual", reporting that HTML validation and math rendering were skipped
because the system 'tidy' is too old and the 'V8' package is unavailable.
That is a property of the local toolchain rather than of the package.)

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

## URL and BugReports

DESCRIPTION deliberately declares no `URL` or `BugReports`. The development
repository is currently private, so advertising it would place URLs on the
package page that resolve to 404 for users. Both fields will be added in the
next release, once the repository is public.
