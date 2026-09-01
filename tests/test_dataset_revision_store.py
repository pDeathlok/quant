from pathlib import Path

from quant.data.dataset_revision_store import (
    DatasetRevisionStore,
    PartitionRevisionInput,
)


def _partition(rows: int, digest: str) -> PartitionRevisionInput:
    return PartitionRevisionInput(row_count=rows, content_sha256=digest * 64)


def test_local_revision_changes_only_when_partition_content_changes(
    tmp_path: Path,
) -> None:
    store = DatasetRevisionStore(metadata_path=tmp_path / "revisions.json")

    first = store.commit(
        "market.daily",
        {"20260831": _partition(5, "a")},
        watermark="20260831",
    )
    same = store.commit(
        "market.daily",
        {"20260831": _partition(5, "a")},
        watermark="20260831",
    )
    corrected = store.commit(
        "market.daily",
        {"20260831": _partition(5, "b")},
        watermark="20260831",
    )

    assert first.revision == 1
    assert first.changed_partitions == ("20260831",)
    assert same.revision == 1
    assert same.changed_partitions == ()
    assert corrected.revision == 2
    assert corrected.changed_partitions == ("20260831",)
    assert store.get("market.daily") == corrected.__class__(
        dataset_id="market.daily",
        revision=2,
        watermark="20260831",
        content_sha256=corrected.content_sha256,
    )


def test_sql_revision_is_transactional_and_partition_aware(tmp_path: Path) -> None:
    store = DatasetRevisionStore(sql_url=f"sqlite:///{tmp_path / 'revisions.db'}")

    first = store.commit(
        "market.daily",
        {
            "20260830": _partition(4, "a"),
            "20260831": _partition(5, "b"),
        },
        watermark="20260831",
    )
    second = store.commit(
        "market.daily",
        {
            "20260830": _partition(4, "c"),
            "20260831": _partition(5, "b"),
        },
        watermark="20260831",
    )

    assert first.revision == 1
    assert first.changed_partitions == ("20260830", "20260831")
    assert second.revision == 2
    assert second.changed_partitions == ("20260830",)
    current = store.get("market.daily")
    assert current is not None
    assert current.revision == 2
    assert current.watermark == "20260831"
