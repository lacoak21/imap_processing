from datetime import datetime
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from spiceypy import utc2et

from imap_processing import imap_module_directory
from imap_processing.cdf.utils import load_cdf, write_cdf
from imap_processing.idex.idex_l1a import PacketParser
from imap_processing.idex.idex_l1b import idex_l1b
from imap_processing.idex.idex_l2a import idex_l2a
from imap_processing.idex.idex_l2b import idex_l2b
from imap_processing.spice.time import et_to_ttj2000ns


@pytest.fixture
def mock_spice_functions():
    """Mock spice functions to avoid loading kernels."""
    with (
        mock.patch("imap_processing.idex.idex_l1b.imap_state") as mock_state,
        mock.patch(
            "imap_processing.idex.idex_l1b.instrument_pointing"
        ) as mock_pointing,
        mock.patch("imap_processing.idex.idex_l1b.solar_longitude") as mock_lon,
    ):
        mock_state.side_effect = lambda t, observer: np.ones((len(t), 6))
        mock_pointing.side_effect = lambda t, instrument, to_frame, cartesian: np.ones(
            (len(t), 3)
        )
        mock_lon.side_effect = lambda t, degrees: np.ones(len(t))

        yield mock_state, mock_pointing, mock_lon


@pytest.fixture
def decom_l2_validation_datasets() -> list[xr.Dataset]:
    """Return a list of ``xarray`` datasets containing test data.

    Returns
    -------
    dataset : list[xarray.Dataset]
        A list of ``xarray`` datasets containing the test data
    """
    path = imap_module_directory / "tests" / "idex" / "test_data" / "l2_validation"
    l2_day1_validation_packet_files = [
        path / "ois_output_12192023_004005",
        path / "ois_output_12192023_011220",
    ]
    l2_day2_validation_packet_files = [
        path / "ois_output_12212023_181601",
        path / "ois_output_12212023_202921",
        path / "ois_output_12212023_213425",
        path / "ois_output_12212023_220641",
        path / "ois_output_12212023_223405",
    ]
    l1a_ds = []
    # l1b_evt = []
    l1a_ds_2 = []
    # l1b_evt_2 = []
    for packet_file in l2_day1_validation_packet_files:
        parsed = PacketParser(packet_file).data
        l1a_ds.append(parsed[0])
        # l1b_evt.append(parsed[1])
    for packet_file in l2_day2_validation_packet_files:
        parsed = PacketParser(packet_file).data
        l1a_ds_2.append(parsed[0])
        # l1b_evt_2.append(parsed[1])
    l1a_day1 = xr.concat(
        l1a_ds, dim="epoch", coords="minimal", compat="override", data_vars="minimal"
    )
    l1a_day2 = xr.concat(
        l1a_ds_2, dim="epoch", coords="minimal", compat="override", data_vars="minimal"
    )
    # l1b_day1 = xr.concat(l1b_evt, dim="epoch")
    # l1b_day2 = xr.concat(l1b_evt_2, dim="epoch")
    write_cdf(l1a_day1)
    write_cdf(l1a_day2)
    # write_cdf(l1b_day1)
    # write_cdf(l1b_day2)


def test_idex_l1b_validation(
    use_fake_spin_data_for_time, imap_ena_sim_metakernel, mock_spice_functions
):
    """Test the L1B validation for IDEX."""
    # Validate the L1B data against expected results.
    day1 = load_cdf(
        "/Users/luco3133/projects/imap_processing/data/imap/idex/idex_validation/imap_idex_l1a_sci-1week_20231219_v999.cdf"
    )
    day2 = load_cdf(
        "/Users/luco3133/projects/imap_processing/data/imap/idex/idex_validation/imap_idex_l1a_sci-1week_20231221_v999.cdf"
    )
    use_fake_spin_data_for_time(start_met=440640776.0, end_met=440893988)
    idex_l1b_ds = idex_l1b(day1)
    idex_l1b_ds2 = idex_l1b(day2)
    write_cdf(idex_l1b_ds)
    write_cdf(idex_l1b_ds2)


def test_idex_l2a_validation(ancillary_files):
    """Test the L2A validation for IDEX."""
    # Validate the L2A data against expected results.
    day1 = load_cdf(
        "/Users/luco3133/projects/imap_processing/data/imap/idex/idex_validation/imap_idex_l1b_sci-1week_20231219_v999.cdf"
    )
    day2 = load_cdf(
        "/Users/luco3133/projects/imap_processing/data/imap/idex/idex_validation/imap_idex_l1b_sci-1week_20231221_v999.cdf"
    )
    idex_l2a_ds = idex_l2a(day1, ancillary_files)
    idex_l2a_ds2 = idex_l2a(day2, ancillary_files)
    write_cdf(idex_l2a_ds)
    write_cdf(idex_l2a_ds2)


def test_idex_l2b_validation(furnish_time_kernels):
    """Test the L2A validation for IDEX."""
    # Validate the L2A data against expected results.
    day1 = load_cdf(
        "/Users/luco3133/projects/imap_processing/data/imap/idex/idex_validation/imap_idex_l2a_sci-1week_20231219_v999.cdf"
    )
    day2 = load_cdf(
        "/Users/luco3133/projects/imap_processing/data/imap/idex/idex_validation/imap_idex_l2a_sci-1week_20231221_v999.cdf"
    )
    times = pd.read_csv(
        "/Users/luco3133/projects/imap_processing/imap_processing/tests/idex/test_data/l2_validation/ON_OFF_DOY_353_355.csv"
    )
    logs = []
    events = [1 if state == "On" else 0 for state in times["state"].values]
    dts = [
        datetime.strptime(date, "%Y/%j-%H:%M:%S.%f").strftime("%Y-%m-%dT%H:%M:%S.%f")
        for date in times["timestamp"].values
    ]
    epoch_vals = [et_to_ttj2000ns(utc2et(dt)) for dt in dts]
    with mock.patch(
        "imap_processing.idex.idex_l2b.get_science_acquisition_timestamps",
        return_value=(logs, epoch_vals, events),
    ):
        l2b_ds, l2c_ds = idex_l2b([day1, day2], [xr.Dataset()])
    write_cdf(l2b_ds)
    write_cdf(l2c_ds)
