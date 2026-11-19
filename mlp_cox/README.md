# MLP-Cox

Use the environment and run sequence in the root README. This stage consumes
the current Cox/RSF outputs and the same frozen folds. The CPU model uses
full-risk-set Efron loss, ReLU and AdamW. The protocol defines the candidate
grid, early stopping, fold-specific seed derivation and outer refit rule.

Seed 11 supplies the manuscript figures. The other two seeds are independent
sensitivity runs. Per-patient predictions, checkpoints and audit records are
written to the local ignored output directory.
