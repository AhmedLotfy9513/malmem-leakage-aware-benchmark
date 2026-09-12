# Benchmark Protocol

## Dataset

CIC-MalMem-2022

Total records:
58,596

Binary distribution:
- Benign: 29,298
- Malware: 29,298

Malware categories:
- Spyware
- Ransomware
- Trojan

Family-level evaluation uses fifteen malware families plus Benign.

## Primary Leakage-Aware Split

Total:
58,596

Development:
46,888

Locked_Test:
11,708

Unique feature-profile groups:
58,027

Development / Locked_Test Group_ID overlap:
0

## Label Definitions

Binary:
Class

Four-class:
Malware_Category

Sixteen-class:
Family

The raw Category column contains sample/dump identifiers and is not
used as the four-class target.

## Audited Label Conflicts

Binary conflicting groups:
0

Four-class category conflicting groups:
9

Sixteen-class family conflicting groups:
19

## Primary Metrics

Binary:
- Accuracy
- Precision
- Recall
- F1
- False Positive Rate
- False Negative Rate

Multiclass:
- Macro-F1 as the primary family-balanced metric
- Accuracy as a secondary metric
- Weighted-F1 when reproducing protocols that use weighted averaging

## Leakage Controls

- Group-aware data partitioning
- Supervised feature selection inside training folds only
- Resampling inside training folds only
- Group-aware inner validation for tuning
- Locked test separation
- Same partitions for compared models

## Interpretability

SHAP:
predicted-class local explanations.

Explanation stability:
multi-scale numerical perturbation analysis.

Counterfactual:
training-range-constrained, prototype-guided model counterfactual.

## ATT&CK Alignment

Evidence chain:

Memory Feature
-> Forensic Evidence
-> Behavioral Hypothesis
-> ATT&CK Candidate
-> Evidence Level
-> Predicted-Class SHAP
-> Candidate Support
-> Statistical Characterization

ATT&CK alignment represents evidence-supported hypotheses, not
confirmed technique detection.