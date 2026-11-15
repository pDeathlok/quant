import json

import numpy as np
import pandas as pd

from quant.features.short_etf_rank import ARTIFACT, MACRO_SCALES, MAX_SHIFT, VERSION
from quant.routine.short_etf_flow import SCHEMA_VERSION, SNAPSHOT_DIRECTORY
from quant.routine.short_etf_rank import market_context_identity, rank_with_etf


def test_rank_uses_one_exact_day_context_and_falls_back_to_base(tmp_path):
    artifact = tmp_path / ARTIFACT
    artifact.parent.mkdir(parents=True)
    coefficients = np.zeros((4, len(MACRO_SCALES)))
    coefficients[2, 1] = 1
    artifact.write_text(json.dumps({
        "schema": VERSION, "market_features": list(MACRO_SCALES), "sides": {"left": {
            "active": True, "trained_through": "2026-08-21", "strength": .5,
            "signals": ["B1"], "base_sha256": "base", "coefficients": coefficients.tolist(),
        }},
    }))
    date = "2026-11-06"
    path = tmp_path / SNAPSHOT_DIRECTORY / "20261106.json"
    path.parent.mkdir(parents=True)
    payload = {
        "status": "success", "schema_version": SCHEMA_VERSION, "signal_date": date,
        "features": {"etf_all_flow_pct_5": .1, "etf_all_turnover_multiple_20": 2},
    }
    path.write_text(json.dumps(payload))
    identity = market_context_identity(date, tmp_path)
    path.write_text(json.dumps({**payload, "observed_at": "later"}))
    assert market_context_identity(date, tmp_path) == identity

    base = np.array([.50, .52])
    flags = pd.DataFrame({"B1": [True, False]})
    adjusted, meta = rank_with_etf("left", flags, base, date, "base", project_root=tmp_path)
    assert meta["status"] == "applied"
    np.testing.assert_allclose(adjusted, [.50 + MAX_SHIFT, .52])
    for other_date, digest in [(date, "different-base"), ("2026-11-05", "base")]:
        score, _ = rank_with_etf("left", flags, base, other_date, digest, project_root=tmp_path)
        np.testing.assert_array_equal(score, base)

    payload["features"]["etf_all_flow_pct_5"] = -.1
    path.write_text(json.dumps(payload))
    assert market_context_identity(date, tmp_path) != identity
