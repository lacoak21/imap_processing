import dataclasses

import numpy as np
import pytest
import xarray as xr

from imap_processing.glows.l1b.glows_l1b import process_de, process_histogram
from imap_processing.glows.l1b.glows_l1b_data import (
    AncillaryParameters,
    DirectEventL1B,
    HistogramL1B,
)


@pytest.fixture()
def hist_dataset():
    variables = {
        "flight_software_version": np.zeros((20,)),
        "seq_count_in_pkts_file": np.zeros((20,)),
        "last_spin_id": np.zeros((20,)),
        "flags_set_onboard": np.zeros((20,)),
        "is_generated_on_ground": np.zeros((20,)),
        "number_of_spins_per_block": np.zeros((20,)),
        "number_of_bins_per_histogram": np.zeros((20,)),
        "number_of_events": np.zeros((20,)),
        "filter_temperature_average": np.zeros((20,)),
        "filter_temperature_variance": np.zeros((20,)),
        "hv_voltage_average": np.zeros((20,)),
        "hv_voltage_variance": np.zeros((20,)),
        "spin_period_average": np.zeros((20,)),
        "spin_period_variance": np.zeros((20,)),
        "pulse_length_average": np.zeros((20,)),
        "pulse_length_variance": np.zeros((20,)),
        "imap_start_time": np.zeros((20,)),
        "imap_time_offset": np.zeros((20,)),
        "glows_start_time": np.zeros((20,)),
        "glows_time_offset": np.zeros((20,)),
    }
    epoch = xr.DataArray(np.arange(20), name="epoch", dims=["epoch"])

    bins = xr.DataArray(np.arange(3600), name="bins", dims=["bins"])

    ds = xr.Dataset(coords={"epoch": epoch})
    ds["histograms"] = xr.DataArray(
        np.zeros((20, 3600)),
        dims=["epoch", "bins"],
        coords={"epoch": epoch, "bins": bins},
    )

    for var in variables:
        ds[var] = xr.DataArray(variables[var], dims=["epoch"], coords={"epoch": epoch})

    return ds


@pytest.fixture()
def de_dataset():
    variables = {
        "seq_count_in_pkts_file": np.zeros((20,)),
        "number_of_de_packets": np.zeros((20,)),
        "imap_sclk_last_pps": np.zeros((20,)),
        "glows_sclk_last_pps": np.zeros((20,)),
        "glows_ssclk_last_pps": np.zeros((20,)),
        "imap_sclk_next_pps": np.zeros((20,)),
        "catbed_heater_active": np.zeros((20,)),
        "spin_period_valid": np.zeros((20,)),
        "spin_phase_at_next_pps_valid": np.zeros((20,)),
        "spin_period_source": np.zeros((20,)),
        "spin_period": np.zeros((20,)),
        "spin_phase_at_next_pps": np.zeros((20,)),
        "number_of_completed_spins": np.zeros((20,)),
        "filter_temperature": np.zeros((20,)),
        "hv_voltage": np.zeros((20,)),
        "glows_time_on_pps_valid": np.zeros((20,)),
        "time_status_valid": np.zeros((20,)),
        "housekeeping_valid": np.zeros((20,)),
        "is_pps_autogenerated": np.zeros((20,)),
        "hv_test_in_progress": np.zeros((20,)),
        "pulse_test_in_progress": np.zeros((20,)),
        "memory_error_detected": np.zeros((20,)),
    }
    epoch = xr.DataArray(np.arange(20), name="epoch", dims=["epoch"])

    per_second = xr.DataArray(np.arange(2295), name="per_second", dims=["per_second"])
    direct_event = xr.DataArray(
        np.arange(4), name="direct_event", dims=["direct_event"]
    )

    de_data = np.zeros((20, 2295, 4))
    de_data[0][0] = [1, 2_000, 3, 0]

    variables["filter_temperature"][0] = 100

    ds = xr.Dataset(coords={"epoch": epoch})

    ds["direct_events"] = xr.DataArray(
        de_data,
        dims=["epoch", "per_second", "direct_event"],
        coords={"epoch": epoch, "per_second": per_second, "direct_event": direct_event},
    )

    for var in variables:
        ds[var] = xr.DataArray(variables[var], dims=["epoch"], coords={"epoch": epoch})

    return ds


@pytest.fixture()
def ancillary_dict():
    dictionary = {
        "description": "Table for conversion/decoding ancillary parameters collected "
        "onboard by IMAP/GLOWS",
        "version": "0.1",
        "date_of_creation_yyyymmdd": "20230527",
        "filter_temperature": {
            "min": -30.0,
            "max": 80.0,
            "n_bits": 8,
            "p01": 0.0,
            "p02": 0.0,
            "p03": 0.0,
            "p04": 0.0,
        },
        "hv_voltage": {
            "min": 0.0,
            "max": 3500.0,
            "n_bits": 12,
            "p01": 0.0,
            "p02": 0.0,
            "p03": 0.0,
            "p04": 0.0,
        },
        "spin_period": {"min": 0.0, "max": 20.9712, "n_bits": 16},
        "spin_phase": {"min": 0.0, "max": 360.0, "n_bits": 16},
        "pulse_length": {
            "min": 0.0,
            "max": 255.0,
            "n_bits": 8,
            "p01": 0.0,
            "p02": 0.0,
            "p03": 0.0,
            "p04": 0.0,
        },
    }
    return dictionary


def test_histogram_mapping():
    time_val = 1111111.11
    # A = 2.318
    # B = 69.5454
    expected_temp = 100

    test_hists = np.zeros((200, 3600))
    # For temp
    encoded_val = expected_temp * 2.318 + 69.5454

    # For now, testing types and number of inputs
    output = tuple(
        dataclasses.asdict(
            HistogramL1B(
                test_hists,
                "test",
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                encoded_val,
                encoded_val,
                encoded_val,
                encoded_val,
                encoded_val,
                encoded_val,
                encoded_val,
                encoded_val,
                time_val,
                time_val,
                time_val,
                time_val,
            )
        ).values()
    )

    assert output[17] == time_val

    # Correctly decoded temperature
    assert output[9] - expected_temp < 0.1


def test_process_histograms(hist_dataset):
    time_val = np.single(1111111.11)
    # A = 2.318
    # B = 69.5454
    expected_temp = 100

    test_hists = np.zeros((200,))
    # For temp
    encoded_val = np.single(expected_temp * 2.318 + 69.5454)

    test_l1b = HistogramL1B(
        test_hists,
        "test",
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        encoded_val,
        encoded_val,
        encoded_val,
        encoded_val,
        encoded_val,
        encoded_val,
        encoded_val,
        encoded_val,
        time_val,
        time_val,
        time_val,
        time_val,
    )

    output = process_histogram(hist_dataset)
    assert len(output) == len(dataclasses.asdict(test_l1b))


def test_process_de(de_dataset, ancillary_dict):
    output = process_de(de_dataset)

    # Output has the same length as non-initvar fields in DirectEventL1B
    assert len(output) == len(dataclasses.fields(DirectEventL1B))

    # Validate timeestamp
    assert np.isclose(output[-2][0].data[0], 1.001)
    # Validate pulse length
    assert output[-1][0].data[0] == 3.0

    ancillary = AncillaryParameters(ancillary_dict)

    expected_temp = ancillary.decode("filter_temperature", 100.0)

    assert np.isclose(output[9].data[0], expected_temp)
