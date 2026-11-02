# XFL-CloudIDS P2-S5-02 | Controlled development comparison

This package runs ONE predefined development-only experiment. It does not re-run/overwrite P2-S5-01.

System 5: Local-only and FEDAVG_LOCALBN (one control experiment to isolate BN running buffer retention).
System 6: M1.1-C1 recipient-local-BN correction (one consistency experiment, same frozen coefficients).

This is NOT evidence of final generalization or guaranteed improvement. Do not open utility/gate/final-test data.

Run from the extracted execution directory using `run_p2_s5_02.ps1`. On technical pass, use `sync_p2_s5_02_github.ps1`.

If any command fails, preserve the log and stop. Do not rerun or tune.
