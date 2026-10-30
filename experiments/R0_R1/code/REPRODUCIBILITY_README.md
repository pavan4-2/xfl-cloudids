# XFL-CloudIDS R0/R1 reproducibility record

This experiment stores **source code, frozen configuration, machine-generated result tables/JSON and the terminal log** in GitHub.

It intentionally does **not** commit development datasets, sealed/final-test data, model checkpoints, `.npz`, `.pt`, `.pth`, or other large/raw artifacts.

Repository layout created by the supplied sync script:

```text
experiments/R0_R1/
  code/
  system5/ or system6/
    results/r0/
    results/r1/
    XFL_R0_R1_SYSTEM5.log or XFL_R0_R1_SYSTEM6.log
    RUN_RECORD.md
```

The execution package is frozen by the supervisor. Students must not edit experiment code or result files before committing them.
