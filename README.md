# Leakage-Aware Memory Malware Analysis Benchmark

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22726638.svg)](https://doi.org/10.5281/zenodo.22726638)

## Overview

This repository provides an enriched, leakage-audited,
explainability- and ATT&CK-aligned reproducibility benchmark
derived from CIC-MalMem-2022.

The original CIC-MalMem-2022 dataset remains attributable to
the Canadian Institute for Cybersecurity (CIC),
University of New Brunswick.

## Classification Tasks

- Binary: Benign vs Malware
- Four-class: Benign, Spyware, Ransomware, Trojan
- Sixteen-class: Benign plus fifteen malware families

## Leakage-Aware Evaluation

Audited statistics:

- Total rows: 58,596
- Unique groups: 58,027
- Development rows: 46,888
- Locked test rows: 11,708
- Development/Locked_Test Group_ID overlap: 0
- Binary conflicting groups: 0
- Four-class conflicting groups: 9
- Sixteen-class conflicting groups: 19

Group-aware partitioning is used to prevent known
feature-identical profiles from crossing the locked split.

## Main Components

- Leakage-audited split manifests
- Binary, four-class, and sixteen-class evaluation
- Feature-selection experiments
- Flat and hierarchical classifiers
- Nested static model fusion
- Repeated statistical evaluation
- Calibration analysis
- SHAP explainability
- Explanation-stability analysis
- Prototype-guided model counterfactual analysis
- Evidence-aware MITRE ATT&CK alignment
- SHA-256 integrity validation

## MITRE ATT&CK

The ATT&CK workflow follows:

Memory Feature -> Forensic Evidence -> Behavioral Hypothesis
-> ATT&CK Candidate -> Evidence Level -> Predicted-Class SHAP
-> Candidate Support -> Statistical Characterization

ATT&CK alignment represents evidence-supported behavioral
hypotheses, not confirmed ATT&CK technique detection.

## Dataset

Original dataset: CIC-MalMem-2022

Provider: Canadian Institute for Cybersecurity,
University of New Brunswick.

The authoritative dataset location and attribution information
are documented in docs/THIRD_PARTY_DATA.md.

The complete original 55-feature matrix is not included in
this GitHub release candidate.

## Reproducibility

The package provides checks for artifact hashes, file sizes,
split counts, group overlap, label conflicts, and accidental
raw-data inclusion.

See docs/REPRODUCIBILITY.md and
docs/BENCHMARK_PROTOCOL.md.

## Claims and Limitations

This repository does not claim:

- zero-day malware detection
- causal interpretation of SHAP values
- confirmed ATT&CK technique detection
- state-of-the-art sixteen-class performance
- physical feasibility of every model counterfactual
- end-to-end operational latency from model timing alone

See docs/CLAIMS_AND_LIMITATIONS.md.

## Citation

Citation metadata is provided in CITATION.cff.

The final manuscript citation and Zenodo DOI will be inserted
after archival publication.

## Version

Current version: 0.1.0

## Release Status

Pre-publication reproducibility release candidate.

GitHub URL: Pending

Zenodo DOI: Pending

Kaggle release: Pending

## License

Original code, scripts, documentation, and repository-authored
materials are released under the Apache License 2.0.

CIC-MalMem-2022 is third-party material and is **not** relicensed
under Apache-2.0 by this repository. Dataset users must follow the
original CIC attribution and citation requirements.

See `LICENSE`, `LICENSE_NOTICE.md`, and
`docs/THIRD_PARTY_DATA.md` for details.

## Archival DOI

A citable archival snapshot of this reproducibility benchmark is
available through Zenodo.

**DOI:** 10.5281/zenodo.22726638

The DOI corresponds to the Zenodo-archived release generated from
the GitHub repository. Version-specific release information is
documented in `CHANGELOG.md`.
