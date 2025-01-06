"""Tests the L1 processing for decommutated IDEX data"""

import numpy as np
import pytest
import xarray as xr
from cdflib.xarray.xarray_to_cdf import ISTPError

from imap_processing import imap_module_directory
from imap_processing.cdf.utils import load_cdf, write_cdf


def test_idex_cdf_file(decom_test_data: xr.Dataset):
    """Verify the CDF file can be created with no errors.

    Parameters
    ----------
    decom_test_data : xarray.Dataset
        The dataset to test with
    """

    file_name = write_cdf(decom_test_data)

    assert file_name.exists()
    assert file_name.name == "imap_idex_l1a_sci_20231218_v001.cdf"


def test_bad_cdf_attributes(decom_test_data: xr.Dataset):
    """Ensure an ``ISTPError`` is raised when using bad CDF attributes.

    Parameters
    ----------
    decom_test_data : xarray.Dataset
        The dataset to test with
    """
    del decom_test_data["TOF_High"].attrs["DEPEND_1"]

    with pytest.raises(ISTPError):
        write_cdf(decom_test_data)


def test_bad_cdf_file_data(decom_test_data: xr.Dataset):
    """Ensure an ``ISTPError`` is raised when using bad data.

    Parameters
    ----------
    decom_test_data : xarray.Dataset
        The dataset to test with
    """
    bad_data_attrs = {
        "CATDESC": "Bad_Data",
        "DEPEND_0": "epoch",
        "DISPLAY_TYPE": "no_plot",
        "FIELDNAM": "Bad_Data",
        "FILLVAL": "",
        "FORMAT": "E12.2",
        "LABLAXIS": "Bad_Data",
        "UNITS": "",
        "VALIDMIN": "1",
        "VALIDMAX": "50",
        "VAR_TYPE": "support_data",
        "VAR_NOTES": """How did this data end up in here?
                        The CDF creation better fail.""",
    }
    bad_data_xr = xr.DataArray(
        name="bad_data",
        data=np.linspace(1, 50, 50),
        dims=("bad_data"),
        attrs=bad_data_attrs,
    )
    decom_test_data["Bad_data"] = bad_data_xr

    with pytest.raises(ISTPError):
        write_cdf(decom_test_data)


def test_idex_tof_high_data_from_cdf(decom_test_data: xr.Dataset):
    """Verify that a sample of the data is correct inside the CDF file.

    ``impact_14_tof_high_data.txt`` has been verified correct by the IDEX team

    Parameters
    ----------
    decom_test_data : xarray.Dataset
        The dataset to test with
    """
    with open(
        f"{imap_module_directory}/tests/idex/test_data/impact_14_tof_high_data.txt"
    ) as f:
        data = np.array([int(line.rstrip()) for line in f])

    file_name = write_cdf(decom_test_data)
    l1_data = load_cdf(file_name)
    assert (l1_data["TOF_High"][13].data == data).all()


def test_validate_l1a_idex_data_variables(
    decom_test_data: xr.Dataset, l1a_example_data: dict
):
    """
    Verify that each of the 6 waveform and telemetry arrays are equal to the
    corresponding array produced by the IDEX team using the same l0 file.

    The comparison is limited to `num_events` because the L1A example contains fewer
    events (due to file size requirements) than the SCD dataset.


    Parameters
    ----------
    decom_test_data : xarray.Dataset
        The dataset to test with
    l1a_example_data: dict
        A dictionary containing the 6 waveform and telemetry arrays
    """
    # Lookup table to match the SDS array names to the Idex Team array names
    l1a_examples = l1a_example_data[0]
    # Number of events in the l1a_examples dict for each data_variable
    num_events = l1a_example_data[1]
    match_variables = {
        "TOF L": "TOF_Low",
        "TOF H": "TOF_High",
        "TOF M": "TOF_Mid",
        "Target H": "Target_High",
        "Target L": "Target_Low",
        "Ion Grid": "Ion_Grid",
        "Time (high sampling)": "time_high_sr",
        "Time (low sampling)": "time_low_sr",
    }

    # The Engineering data is converting to UTC, and the SDC is converting to J2000,
    # for 'epoch' and 'Timestamp' so this test is using the raw time value 'SCHOARSE' to
    # validate time
    arrays_to_skip = ["Timestamp", "Epoch"]

    # loop through all keys from the l1a example dict
    for var in l1a_examples.keys():
        if var not in arrays_to_skip:
            # Find the corresponding array name
            if var in match_variables.keys():
                cdf_var = match_variables[var]
            else:
                cdf_var = var.lower()

            assert np.array_equal(
                decom_test_data[cdf_var][:num_events], l1a_examples[var]
            ), (
                f"The array '{cdf_var}' does not equal the expected example array "
                f"'{var}' produced by the IDEX team"
            )
