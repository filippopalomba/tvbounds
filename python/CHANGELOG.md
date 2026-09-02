# tvbounds 0.1.2

Initial release.

* `tvbounds_attrition()`: sensitivity bounds and bootstrap inference for treatment
  effects in randomized experiments with attrition, under total-variation or
  contamination neighborhoods, with optional covariate-pooled bounds.
* `tvbounds_counterfactual()`: bounds on counterfactual predictions in
  structural models under general divergence neighborhoods, through an
  interface to Julia and the Artelys KNITRO solver.
* `tvbounds_riv()`: bounds and breakdown budgets for recentered instrumental
  variables under total-variation or contamination neighborhoods.
* `tvbounds_plot()` and `tvbounds_summary()`: sensitivity-bounds plots and
  the paper's summary measures (breakdown budget, shadow price, exchange
  rate, equivalent sample size), with `.plot()` and `.summary()` methods.
