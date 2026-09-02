# Reconstruction Notes

## Why this file exists

The F0-R1 Student Execution Manual refers to `XFL-CloudIDS_F0-R1_Execution_Package.zip`, but that original ZIP was not available when this package was reconstructed. The implementation therefore separates protocol values explicitly frozen by the manual from low-level implementation defaults that had to be reconstructed.

## Manual-frozen protocol values

The following are directly specified by the F0-R1 manual and are represented in `f0_config.json`:

- amendment `F0-R1-A1`
- CIC-IDS2017 real CSVs
- row cap 750,000 with rare-preserving proportional sampling
- five tenants
- seeds 17 and 29
- Dirichlet alpha 0.3 and 0.1
- split fractions 60% train / 15% utility / 10% gate / 15% sealed test
- LayerNorm MLP, hidden sizes 128 and 64, dropout 0.20
- batch 1024, learning rate 0.001, weight decay 0.0001
- two local epochs, ten rounds, decision rounds 5-9
- negative-transfer margin 0.005 and material-loss margin 0.02 macro-F1
- reliability gamma grid 0.80, 0.90, 0.95
- bootstrap count 1,000
- risk budget 0.20, CVaR lambda 0.25, q 0.80
- gate tolerances: F1 margin 0, FAR 0.005, worst-class recall 0.01
- nontrivial change: prediction change >=0.001 plus external weight mass >=0.01
- the R1.1-R1.6 decision criteria in Section 9 of the manual

## Reconstructed implementation defaults

The manual does not provide every low-level value needed to implement executable code. The following defaults are therefore declared rather than hidden:

- `rare_preserve_max_count = 2500`: operational threshold used to identify classes preserved in full before proportional allocation of the remaining row cap.
- `fedprox_mu = 0.01`: proximal coefficient for the F0-R1 FedProx screening baseline.
- `utility_windows = 3`: number of ordered utility sub-windows used when no real-run override is allowed.
- `utility_min_examples = 1`: minimum observations for an F0-R1 utility row to be marked eligible. The manual's primary requirement is that eligible rare-class rows exist; it does not state a numerical minimum.
- `utility_success_threshold = 0.0`: positive transfer utility defines a historical success for reliability updating.
- `reliability_lower_quantile = 0.05`: lower 5% Beta posterior quantile used as the conservative reliability score.
- `temporal_mu = 0.05`: temporal weight-stability penalty in the minimal RUTRA optimization.
- optimizer maximum iterations 200 and tolerance 1e-7.

These values must **not** be represented as values recovered from the missing original ZIP. If the supervisor possesses the original implementation specification for any of them, replace the reconstructed value before the controlled research run, record the amendment/provenance, and regenerate the configuration hash.

## Method boundary preserved

The implementation follows the project's finalized RUTRA-Gate boundary:

- measured recipient/class/window transfer utility is the collaboration signal;
- current-round cosine similarity is logged only as a diagnostic for the pFedGraph-core screening comparison and does not replace measured utility in RUTRA-Gate;
- reliability is based on prior evidence and selected prospectively;
- recipient weights are constrained by reliability risk and include tail-risk/temporal terms;
- final deployment uses a held-out temporal bootstrap gate against the local reference;
- optimizer failure falls back to the local reference.

FedFomo and pFedGraph outputs in R1.5 are labelled **screening diagnostics**, matching the F0-R1 manual; they are not publication-faithful final baselines.

## Verification performed before distribution

The reconstructed code was syntax-checked and its R1.1-R1.5 synthetic smoke path was executed end-to-end in the artifact-generation environment. That environment was **not** the frozen research environment (it used Python 3.13 and CPU PyTorch), so this check demonstrates code-path integration only. Students must repeat the smoke and the real run in the exact F0-R1 environment required by the manual.
