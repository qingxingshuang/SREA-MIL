# SREA-MIL

Research code for State-Relation Evidence Aggregation MIL for cervical cytology
whole-slide classification. The implementation, protocol configuration, and
usage instructions are in [SREA_MIL_Code_Release_20260927](SREA_MIL_Code_Release_20260927/README.md).

```bash
cd SREA_MIL_Code_Release_20260927
pip install -e .[test]
python scripts/fetch_dtfd.py
pytest -q
python scripts/smoke_test.py
```

The cohort, extracted features, exact splits, and trained weights are not
distributed. The included toy-data generator checks execution only, not the
manuscript's reported performance. The official DTFD-MIL dependency is fetched
separately from a pinned upstream commit and is not distributed in this
repository. See the code package's
[citation](SREA_MIL_Code_Release_20260927/CITATION.md),
[license status](SREA_MIL_Code_Release_20260927/LICENSE_PENDING.md), and
[third-party notices](SREA_MIL_Code_Release_20260927/THIRD_PARTY_NOTICES.md).
