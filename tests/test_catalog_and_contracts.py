from __future__ import annotations

import pandas as pd
import pytest

from seismic_bench import catalog as cat
from seismic_bench.contracts import CATALOG_CONTRACT, Contract, Rule


def _row(**overrides) -> dict:
    base = {
        "event_id": "usgs:test1",
        "time": "2020-01-01T00:00:00Z",
        "latitude": 35.0,
        "longitude": -119.0,
        "depth_km": 8.0,
        "magnitude": 4.2,
        "mag_type": "Mw",
        "source": "usgs",
    }
    base.update(overrides)
    return base


class TestConform:
    def test_produces_the_canonical_schema(self):
        out = cat.conform(pd.DataFrame([_row()]))
        assert list(out.columns) == cat.COLUMNS
        assert str(out["time"].dtype) == "datetime64[ns, UTC]"
        assert out["mag_type"].iloc[0] == "mw", "magnitude type should be lowercased"

    def test_missing_columns_raise_rather_than_fill(self):
        """Inventing a missing column would hide a broken ingester."""
        bad = pd.DataFrame([_row()]).drop(columns=["depth_km"])
        with pytest.raises(ValueError, match="missing columns"):
            cat.conform(bad)

    def test_longitude_is_wrapped_to_signed_range(self):
        out = cat.conform(pd.DataFrame([_row(longitude=350.0), _row(longitude=190.0)]))
        assert out["longitude"].tolist() == [-10.0, -170.0]

    def test_output_is_time_sorted(self):
        rows = [
            _row(event_id="b", time="2020-06-01T00:00:00Z"),
            _row(event_id="a", time="2020-01-01T00:00:00Z"),
        ]
        out = cat.conform(pd.DataFrame(rows))
        assert out["event_id"].tolist() == ["a", "b"]


class TestDeduplicate:
    def test_keeps_the_latest_revision(self):
        """Catalogs get revised; a re-ingest should improve, not duplicate."""
        rows = [
            _row(event_id="x", time="2020-01-01T00:00:00Z", magnitude=4.0),
            _row(event_id="x", time="2020-01-02T00:00:00Z", magnitude=4.5),
        ]
        out = cat.deduplicate(cat.conform(pd.DataFrame(rows)))
        assert len(out) == 1
        assert out["magnitude"].iloc[0] == 4.5

    def test_distinct_ids_are_preserved(self):
        rows = [_row(event_id="x"), _row(event_id="y")]
        assert len(cat.deduplicate(cat.conform(pd.DataFrame(rows)))) == 2


class TestCatalogContract:
    def _apply(self, rows: list[dict]):
        return CATALOG_CONTRACT.apply(cat.conform(pd.DataFrame(rows)))

    def test_valid_rows_pass_through(self):
        clean, quarantined = self._apply([_row()])
        assert len(clean) == 1 and quarantined.empty

    @pytest.mark.parametrize(
        ("override", "reason"),
        [
            ({"latitude": 120.0}, "latitude_out_of_range"),
            ({"magnitude": -9.9}, "magnitude_implausible"),
            ({"magnitude": None}, "magnitude_missing"),
            ({"depth_km": 5000.0}, "depth_implausible"),
            ({"latitude": 0.0, "longitude": 0.0}, "null_island"),
            ({"time": "not-a-date"}, "time_unparseable"),
        ],
    )
    def test_bad_rows_are_quarantined_with_a_reason(self, override, reason):
        clean, quarantined = self._apply([_row(**override)])
        assert clean.empty
        assert len(quarantined) == 1
        assert reason in quarantined["_reasons"].iloc[0]

    def test_a_bad_row_does_not_block_good_ones(self):
        """The whole reason for quarantine rather than fail-fast."""
        clean, quarantined = self._apply(
            [_row(event_id="ok1"), _row(event_id="bad", magnitude=-9.9), _row(event_id="ok2")]
        )
        assert clean["event_id"].tolist() == ["ok1", "ok2"]
        assert len(quarantined) == 1

    def test_all_broken_rules_are_reported_not_just_the_first(self):
        clean, quarantined = self._apply([_row(magnitude=None, depth_km=5000.0)])
        assert clean.empty
        reasons = quarantined["_reasons"].iloc[0]
        assert "magnitude_missing" in reasons and "depth_implausible" in reasons

    def test_negative_depth_is_legitimate(self):
        """Events above sea level are real. Rejecting them would be a science bug."""
        clean, quarantined = self._apply([_row(depth_km=-2.5)])
        assert len(clean) == 1 and quarantined.empty

    def test_missing_depth_is_tolerated(self):
        """Some agencies cannot constrain depth. That is not a reason to drop the event."""
        clean, quarantined = self._apply([_row(depth_km=None)])
        assert len(clean) == 1 and quarantined.empty

    def test_real_sample_is_clean(self, sample_catalog):
        clean, quarantined = CATALOG_CONTRACT.apply(sample_catalog)
        assert quarantined.empty
        assert len(clean) == len(sample_catalog)

    def test_empty_input_is_handled(self):
        clean, quarantined = CATALOG_CONTRACT.apply(cat.empty())
        assert clean.empty and quarantined.empty


class TestContractMechanics:
    def test_custom_contract_splits_as_expected(self):
        contract = Contract(
            table="toy",
            rules=(Rule("must_be_positive", lambda d: d["v"] > 0),),
        )
        clean, quarantined = contract.apply(pd.DataFrame({"v": [1, -1, 2]}))
        assert clean["v"].tolist() == [1, 2]
        assert quarantined["v"].tolist() == [-1]
        assert quarantined["_reasons"].tolist() == ["must_be_positive"]

    def test_quarantine_file_is_written(self, tmp_path):
        contract = Contract(table="toy", rules=(Rule("positive", lambda d: d["v"] > 0),))
        _, quarantined = contract.apply(pd.DataFrame({"v": [-1]}))
        path = contract.write_quarantine(quarantined, root=tmp_path)
        assert path is not None and path.exists()
        assert pd.read_parquet(path)["_reasons"].tolist() == ["positive"]

    def test_nothing_written_when_nothing_quarantined(self, tmp_path):
        contract = Contract(table="toy", rules=(Rule("positive", lambda d: d["v"] > 0),))
        _, quarantined = contract.apply(pd.DataFrame({"v": [1]}))
        assert contract.write_quarantine(quarantined, root=tmp_path) is None
