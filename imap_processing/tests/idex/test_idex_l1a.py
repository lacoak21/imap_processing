"""Tests the L1 processing for decommutated IDEX data"""

import numpy as np
import pytest
import xarray as xr
from cdflib.xarray.xarray_to_cdf import ISTPError

from imap_processing import imap_module_directory
from imap_processing.cdf.utils import load_cdf, write_cdf
from imap_processing.tests.idex.test_utils import load_hdf_file


@pytest.fixture()
def l1a_example_data():
    """
    Pytest fixture to load example L1A data (produced by the IDEX team) for testing.

    Returns
    -------
    dict
      A dictionary containing the 6 waveform and telemetry arrays
    """
    l1a_test_path = (
        f"{imap_module_directory}/tests/idex/validation_files/L1A_Example.h5"
    )
    # The H5 files were modified to include only the first 10 events to ensure the file
    # size stays below the maximum limit of 1MB.
    num_events = 7
    return load_hdf_file(l1a_test_path, num_events=num_events), num_events


def test_idex_cdf_file(decom_test_data: xr.Dataset):
    """Verify the CDF file can be created with no errors.

    Parameters
    ----------
    decom_test_data : xarray.Dataset
        The dataset to test with
    """

    file_name = write_cdf(decom_test_data)

    assert file_name.exists()
    assert file_name.name == "imap_idex_l1a_sci_20231214_v001.cdf"


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

    file_name = write_cdf(decom_test_data, istp=False)
    l1_data = load_cdf(file_name)
    assert (l1_data["TOF_High"][13].data == data).all()


def test_validate_l1a_idex_data_variables(
    decom_test_data: xr.Dataset, l1a_example_data: dict
):
    """
    Verify that each of the 6 waveform and telemetry arrays are equal to the
    corresponding array produced by the IDEX team using the same l0 file.

    Parameters
    ----------
    decom_test_data : xarray.Dataset
        The dataset to test with
    l1a_example_data: dict
        A dictionary containing the 6 waveform and telemetry arrays
    """
    # Lookup table to match the SDS array names to the Idex Team array names
    waveform_variables = {
        "TOF_Low": "TOF L",
        "TOF_High": "TOF H",
        "TOF_Mid": "TOF M",
        "Target_High": "Target H",
        "Target_Low": "Target L",
        "Ion_Grid": "Ion Grid",
    }

    for var in decom_test_data.data_vars:
        l1a_examples = l1a_example_data[0]
        num_events = l1a_example_data[1]
        if var in waveform_variables.keys():
            eng_var = waveform_variables[var]
        else:
            eng_var = var.upper()

        assert np.array_equal(
            decom_test_data[var][:num_events], l1a_examples[eng_var]
        ), (
            f"The array '{var}' does not equal the expected example array '{eng_var}' "
            f"produced by the IDEX team"
        )
