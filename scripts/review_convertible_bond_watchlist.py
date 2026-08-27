#!/usr/bin/env python3
"""Plan or apply manual convertible-bond allotment watchlist changes."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quant.routine.convertible_bond_watchlist_review import main  # noqa: E402


def _shared_daily_allotment_workspace(**kwargs):
    """Delegate refreshes to the same entry used by the page and daily update."""

    from quant.webapp.services import get_convertible_bond_allotments

    return get_convertible_bond_allotments(**kwargs)


def _sync_watchlist_scores():
    """Refresh the lightweight score cache used by the watchlist page."""

    from quant.webapp.services import refresh_similar_pattern_watchlist_scores

    return refresh_similar_pattern_watchlist_scores()


if __name__ == "__main__":
    raise SystemExit(
        main(
            workspace_loader=_shared_daily_allotment_workspace,
            data_synchronizer=_sync_watchlist_scores,
        )
    )
