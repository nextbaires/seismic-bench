from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seismic_bench.bench.grid import CALIFORNIA, EARTH_RADIUS_KM, Grid


class TestGeometry:
    def test_cell_counts(self):
        grid = Grid(lon_min=0.0, lon_max=10.0, lat_min=0.0, lat_max=5.0, cell_deg=1.0)
        assert (grid.n_lon, grid.n_lat, grid.n_cells) == (10, 5, 50)
        assert grid.shape == (5, 10)

    def test_centers_are_offset_by_half_a_cell(self):
        grid = Grid(lon_min=0.0, lon_max=2.0, lat_min=0.0, lat_max=2.0, cell_deg=1.0)
        lon, lat = grid.cell_centers()
        assert lon.tolist() == [0.5, 1.5, 0.5, 1.5]
        assert lat.tolist() == [0.5, 0.5, 1.5, 1.5]

    def test_centers_are_lat_major(self):
        """Flattening order must match `bin_events`, or forecasts are transposed."""
        grid = Grid(lon_min=0.0, lon_max=3.0, lat_min=0.0, lat_max=2.0, cell_deg=1.0)
        lon, lat = grid.cell_centers()
        assert len(lon) == grid.n_cells
        # First n_lon entries share the lowest latitude band.
        assert np.allclose(lat[: grid.n_lon], lat[0])

    def test_total_area_matches_the_sphere(self):
        """A whole-Earth grid must sum to 4*pi*R^2."""
        grid = Grid(lon_min=-180.0, lon_max=180.0, lat_min=-90.0, lat_max=90.0, cell_deg=10.0)
        expected = 4.0 * np.pi * EARTH_RADIUS_KM**2
        assert grid.cell_areas_km2().sum() == pytest.approx(expected, rel=1e-9)

    def test_cells_shrink_toward_the_poles(self):
        grid = Grid(lon_min=0.0, lon_max=10.0, lat_min=0.0, lat_max=80.0, cell_deg=10.0)
        areas = grid.cell_areas_km2().reshape(grid.shape)
        band_areas = areas[:, 0]
        assert np.all(np.diff(band_areas) < 0)

    def test_rejects_invalid_bounds(self):
        with pytest.raises(ValueError, match="increasing"):
            Grid(lon_min=10.0, lon_max=0.0, lat_min=0.0, lat_max=5.0, cell_deg=1.0)
        with pytest.raises(ValueError, match="positive"):
            Grid(lon_min=0.0, lon_max=10.0, lat_min=0.0, lat_max=5.0, cell_deg=0.0)


class TestIndexing:
    @pytest.fixture
    def grid(self) -> Grid:
        return Grid(lon_min=0.0, lon_max=4.0, lat_min=0.0, lat_max=4.0, cell_deg=1.0)

    def test_known_points(self, grid):
        # (lon=0.5, lat=0.5) -> ix=0, iy=0 -> flat 0
        # (lon=3.5, lat=3.5) -> ix=3, iy=3 -> flat 15
        idx = grid.index_of(np.array([0.5, 3.5]), np.array([0.5, 3.5]))
        assert idx.tolist() == [0, 15]

    def test_points_outside_return_minus_one(self, grid):
        idx = grid.index_of(np.array([-1.0, 5.0, 2.0]), np.array([2.0, 2.0, -1.0]))
        assert idx.tolist() == [-1, -1, -1]

    def test_upper_edge_is_exclusive(self, grid):
        """Half-open cells: a point exactly on the region's far edge is outside.

        Without this, two adjacent regions both claim a boundary event.
        """
        assert grid.index_of(np.array([4.0]), np.array([2.0]))[0] == -1
        assert grid.index_of(np.array([0.0]), np.array([0.0]))[0] == 0


class TestBinning:
    def test_counts_land_in_the_right_cells(self):
        grid = Grid(lon_min=0.0, lon_max=2.0, lat_min=0.0, lat_max=2.0, cell_deg=1.0)
        events = pd.DataFrame({"longitude": [0.5, 0.5, 1.5], "latitude": [0.5, 0.5, 1.5]})
        assert grid.bin_events(events).tolist() == [2.0, 0.0, 0.0, 1.0]

    def test_events_outside_are_dropped_silently(self):
        grid = Grid(lon_min=0.0, lon_max=2.0, lat_min=0.0, lat_max=2.0, cell_deg=1.0)
        events = pd.DataFrame({"longitude": [0.5, 99.0], "latitude": [0.5, 99.0]})
        counts = grid.bin_events(events)
        assert counts.sum() == 1.0

    def test_empty_input_gives_zeros(self):
        grid = Grid(lon_min=0.0, lon_max=2.0, lat_min=0.0, lat_max=2.0, cell_deg=1.0)
        counts = grid.bin_events(pd.DataFrame({"longitude": [], "latitude": []}))
        assert counts.shape == (grid.n_cells,)
        assert counts.sum() == 0.0

    def test_total_is_conserved_on_real_data(self, sample_catalog):
        inside = CALIFORNIA.contains(sample_catalog).sum()
        assert CALIFORNIA.bin_events(sample_catalog).sum() == inside

    def test_contains_matches_indexing(self, sample_catalog):
        mask = CALIFORNIA.contains(sample_catalog)
        assert mask.sum() > 0
        assert len(mask) == len(sample_catalog)
