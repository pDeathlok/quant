from quant.routine import pipeline
from quant.webapp import services


def test_daily_web_workspaces_refreshes_allotment_and_similar_pattern(monkeypatch) -> None:
    calls = []

    monkeypatch.setattr(
        services,
        "get_convertible_bond_allotments",
        lambda **kwargs: calls.append(("allotment", kwargs["refresh"]))
        or {"generated_at": "2026-07-13T08:30:00", "records": []},
    )
    monkeypatch.setattr(
        services,
        "refresh_similar_pattern_analysis",
        lambda: calls.append(("similar", True))
        or {"generated_at": "2026-07-13T08:31:00", "results": []},
    )

    result = pipeline.refresh_daily_web_workspaces()

    assert calls == [("allotment", True), ("similar", True)]
    assert result["convertible_bond_allotments"]["status"] == "success"
    assert result["similar_patterns"]["status"] == "success"
