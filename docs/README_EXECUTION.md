# Reproducibility Execution Guide

## Scope

This package accompanies an enriched, leakage-audited,
explainability- and ATT&CK-aligned benchmark derived from
CIC-MalMem-2022.

The original CIC-MalMem-2022 feature matrix is not redistributed
through this package.

## Recommended Execution Order

1. Prepare the original CIC-MalMem-2022 dataset locally.
2. Run the leakage-audit and group construction workflow.
3. Reconstruct or verify the locked Development / Locked_Test split.
4. Execute binary benchmark experiments.
5. Execute four-class experiments.
6. Execute sixteen-family flat, hierarchical, and fusion experiments.
7. Execute SHAP analysis.
8. Execute counterfactual and explanation-stability analysis.
9. Execute evidence-aware ATT&CK alignment.
10. Run package validation before public release.

## Label Semantics

Binary task:
- Class

Four-class task:
- Malware_Category

Sixteen-class task:
- Family

The original Category field contains sample/dump identifiers and is
not used as the four-class classification target.

## Leakage Controls

Group_ID is used to keep feature-identical or duplicated profiles
within the same partition.

Expected locked split:

- Total rows: 58,596
- Development rows: 46,888
- Locked_Test rows: 11,708
- Unique groups: 58,027
- Development / Locked_Test group overlap: 0

Expected audited label conflicts:

- Binary conflicts: 0
- Four-class category conflicts: 9
- Sixteen-class family conflicts: 19

## Scientific Guardrails

The rigorous evaluation protocol must not be replaced by conventional
random splitting solely to obtain higher scores.

Supervised feature selection and resampling must be restricted to
training folds.

SHAP attribution explains model behavior and does not establish
causality.

Counterfactual examples are model counterfactuals and are not guaranteed
physical malware interventions.

ATT&CK alignment represents evidence-supported behavioral hypotheses
and does not constitute confirmed ATT&CK technique detection.

## Public Release

Before publication on GitHub, Kaggle, Zenodo, or another repository,
verify the authoritative redistribution and licensing conditions of
CIC-MalMem-2022.

Do not describe this package as a newly collected malware dataset unless
new raw memory acquisitions are independently collected.