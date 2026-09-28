# SREA-MIL

Research implementation of State-Relation Evidence Aggregation MIL for
cervical cytology whole-slide classification. The code is hosted at
https://github.com/qingxingshuang/SREA-MIL.

This package reproduces the method and experiment interface described in the
manuscript. It includes the complete SREA-MIL architecture, the matched
pseudo-bag baseline, component ablations, deterministic two-stream feature
loading, five-inner-fold epoch selection, full-development refitting, and
held-out evaluation with Accuracy, Sensitivity, Specificity, and AUROC.

## What is included

- An adapter for separately obtained, pinned official DTFD-MIL first-tier modules.
- Learnable evidence tokenization into eight soft evidence slots.
- Gated state aggregation over learned slot-index order.
- Four-anchor relation aggregation without explicit all-pair slot attention.
- Equal-weight state/relation logit fusion and training-only mutual KL.
- Shared processing of normal and suspected-abnormal feature streams.
- Baseline and ablation configurations used for component analysis.

## What is not included

- Patient data, WSI files, candidate images, or extracted feature arrays.
- Private slide identifiers, center identifiers, or the manuscript split file.
- Trained checkpoints or pretrained weights.
- Server addresses, credentials, absolute research paths, or experiment logs.

Exact table values require the same private cohort, frozen upstream features,
and split assignments. The supplied toy-data generator verifies installation;
it is not intended to reproduce manuscript performance.

## Installation

After cloning the repository, enter the code package directory. Create a
Python 3.10 or newer environment, then run:

    cd SREA_MIL_Code_Release_20260927
    pip install -e .
    python scripts/fetch_dtfd.py

The fetch script obtains `Model/network.py`, `Model/Attention.py`, and the
upstream MIT `LICENSE` from commit `10964f4dcc27c65ce110a0e9a3b9240bff58da8a`, verifies their
SHA-256 hashes, and places them in a user-local cache outside this repository.
The DTFD-MIL source is not distributed in this repository. If automatic
downloading is unavailable, set `SREA_DTFD_ROOT` to a directory containing the
pinned upstream files and `LICENSE`. Do not run arbitrary unverified forks of the
dependency.

For tests:

    pip install -e .[test]
    python scripts/fetch_dtfd.py
    pytest -q

## Quick installation check

    python scripts/smoke_test.py
    python scripts/make_toy_data.py --output toy_data

For a short end-to-end toy run, use the supplied smoke configuration:

    python scripts/run_protocol.py --manifest toy_data/manifest.csv --splits toy_data/splits.json --config configs/smoke_protocol.json --model srea_mil --output outputs/toy_srea

## Preparing data

Each slide is represented by up to two precomputed NumPy feature files:

- suspected-abnormal candidate features;
- normal-candidate features.

Each array must have shape N by 1280. Create a CSV using
data/manifest.example.csv and a de-identified split JSON following
data/splits.example.json. See data/README.md for the exact interface.

## Paper protocol

The default configuration in configs/paper_protocol.json uses:

- seed 42;
- Adam, learning rate 1e-4, weight decay 1e-4;
- batch size 4 and gradient clipping at 5;
- at most 512 features per stream;
- five inner folds with validation AUROC checkpoint selection;
- early stopping patience 15 and maximum 100 epochs;
- median selected epoch count for one full-development refit;
- one held-out evaluation at threshold 0.5.

Run the full protocol with:

    python scripts/run_protocol.py --manifest /path/to/manifest.csv --splits /path/to/splits.json --model srea_mil --output outputs/srea_mil

Generated output directories may contain checkpoints. Keep them out of commits;
the supplied .gitignore excludes common model, data, and output formats.

## Released configurations

- dtfd_mil: official attention-based second tier.
- tokenizer_mil: evidence tokenizer plus mean readout.
- state_mil: direct group projection plus gated state aggregation.
- relation_mil: direct group projection plus inducing-anchor aggregation.
- token_state_mil: tokenizer plus state branch.
- token_relation_mil: tokenizer plus relation branch.
- srea_mil_nomcr: complete dual branch without mutual consistency regularization.
- srea_mil: complete manuscript method.

## Repository layout

    srea_mil/          model, official dependency adapter, data, metrics, and training engine
    scripts/           dependency fetch, training, evaluation, toy-data, and smoke-test tools
    configs/           manuscript protocol configuration
    data/              public manifest and split schemas only
    tests/             deterministic unit and gradient tests

The separately downloaded DTFD-MIL files live in the user-local cache, not
under this repository layout.

## Code availability, licensing, and citation

The implementation and example data interface are available in this repository.
Patient data, extracted features, exact manuscript splits, and trained weights
are not distributed; the example data and smoke test demonstrate execution,
not reproduction of the reported scores. See CITATION.md for a provisional
manuscript reference until bibliographic details are available.

No license has yet been granted for the original SREA-MIL code; see
LICENSE_PENDING.md. Public visibility alone does not grant redistribution or
modification rights. DTFD-MIL remains available from its upstream repository
under its MIT license; see THIRD_PARTY_NOTICES.md. No upstream source files
are included in this release.
