# Legacy notebooks (superseded)

These notebooks belong to an earlier version of the analysis. They are kept for
transparency only. **They do not reproduce the published results and should not be used.**

An internal audit found that the models in these notebooks received row, well and date
identifiers among their inputs, in addition to the environmental predictors, and that some
reported numbers came from separate runs rather than one pipeline. Identifiers and dates let
a tree model recognize a well or a period directly, which inflates the scores of the
dependence-preserving designs.

The analysis in `scripts/` replaces them: one declared set of 37 predictors with identifiers
and dates excluded, physical well keys, monthly climate matched to each record's own
month, preprocessing fitted on training data only, five designs with baselines on identical
partitions, and every number and figure written from saved outputs.
