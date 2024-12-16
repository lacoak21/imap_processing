"""IMAP-IDEX L1B processing module."""

import logging
from enum import Enum

import pandas as pd
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.utils import convert_raw_to_eu

logger = logging.getLogger(__name__)


class ConvFactors(float, Enum):
    """Enum class for conversion factor values."""

    TOF_High = 2.89e-4
    TOF_Low = 5.14e-4
    TOF_Mid = 1.13e-2
    Target_Low = 1.58e1
    Target_High = 1.63e-1
    Ion_Grid = 7.46e-4


def idex_l1b(l1a_dataset: xr.Dataset, data_version: str) -> xr.Dataset:
    """
    Will process IDEX l1a data to create l1b data products.

    Parameters
    ----------
    l1a_dataset : xarray.Dataset
        IDEX L1a dataset to process.
    data_version : str
        Version of the data product being created.

    Returns
    -------
    l1b_dataset : xarray.Dataset
        The``xarray`` dataset containing the science data and supporting metadata.
    """
    logger.info(
        f"Running IDEX L1B processing on dataset: {l1a_dataset.attrs['Logical_source']}"
    )

    # create the attribute manager for this data level
    idex_attrs = ImapCdfAttributes()
    idex_attrs.add_instrument_global_attrs(instrument="idex")
    idex_attrs.add_instrument_variable_attrs(instrument="idex", level="l1b")
    idex_attrs.add_global_attribute("Data_version", data_version)

    var_information_path = (
        f"{imap_module_directory}/idex/idex_variable_unpacking_and_eu_conversion.csv"
    )
    # Read in csv that contains instrument variable setting information
    var_information_df = pd.read_csv(var_information_path)

    processed_vars = unpack_instrument_settings(
        l1a_dataset, var_information_df, idex_attrs
    )

    waveforms_converted = convert_waveforms(l1a_dataset, idex_attrs)

    epoch_da = xr.DataArray(
        l1a_dataset["epoch"],
        name="epoch",
        dims=["epoch"],
        attrs=idex_attrs.get_variable_attributes("epoch"),
    )
    # Create l1b Dataset
    l1b_dataset = xr.Dataset(
        data_vars=processed_vars | waveforms_converted,
        attrs=idex_attrs.get_global_attributes("imap_idex_l1b_sci"),
    )
    # Convert variables
    l1b_dataset = convert_raw_to_eu(
        l1b_dataset,
        conversion_table_path=var_information_path,
        packet_name=var_information_df["packetName"].to_list(),
    )
    vars_to_copy = ["shcoarse", "shfine", "time_high_sr", "time_low_sr"]
    # Copy arrays from the l1a_dataset that do not need l1b processing
    for var in vars_to_copy:
        l1b_dataset[var] = l1a_dataset[var].copy()

    l1b_dataset["epoch"] = epoch_da

    # TODO: Add TriggerMode and TriggerLevel attr

    logger.info("IDEX L1B science data processing completed.")

    return l1b_dataset


def unpack_instrument_settings(
    l1a_dataset: xr.Dataset,
    var_information_df: pd.DataFrame,
    idex_attrs: ImapCdfAttributes = None,
) -> dict[str, xr.DataArray]:
    """
    Unpack raw telemetry data from the l1a dataset into individual variables.

    Parameters
    ----------
    l1a_dataset : xarray.Dataset
        IDEX L1a dataset containing the 6 waveform arrays.
    var_information_df : pd.DataFrame
        Pandas data frame that contains information about each variable
        (e.g., bit-size, starting bit, and padding). This is used to unpack raw
        telemetry data from the input dataset (`l1a_dataset`).
    idex_attrs : ImapCdfAttributes
        CDF attribute manager object.

    Returns
    -------
    telemetry_das : dict
        A dictionary where the keys are the instrument setting array names and the
        values are the unpacked xr.DataArrays.
    """
    telemetry_das = {}

    for _, row in var_information_df.iterrows():
        var_name = row["mnemonic"]

        # Create binary mask of the size of the variable in bits
        mask = (1 << row["unsigned_nbits"]) - 1
        # Determine the number of bits to shift
        shift = row["starting_bit"] - row["nbits_padding_before"]
        # Get the unpacked value by shifting the data to align the desired bits with
        # the least significant bits and applying the mask to isolate the target bits
        unpacked_val = (l1a_dataset[row["packetName"]].data >> shift) & mask

        attrs = idex_attrs.get_variable_attributes(var_name) if idex_attrs else None

        telemetry_das[var_name] = xr.DataArray(
            name=var_name,
            data=unpacked_val,
            dims=("epoch"),
            attrs=attrs,
        )

    return telemetry_das


def convert_waveforms(
    l1a_dataset: xr.Dataset, idex_attrs: ImapCdfAttributes
) -> dict[str, xr.DataArray]:
    """
    Apply transformation from raw dn to picocoulombs (pC) for each of the six waveforms.

    Parameters
    ----------
    l1a_dataset : xarray.Dataset
        IDEX L1a dataset containing the six waveform arrays.
    idex_attrs : ImapCdfAttributes
        CDF attribute manager object.

    Returns
    -------
    waveforms_converted : dict
        A dictionary where the keys are the waveform array names and the values are
        xr.DataArrays representing the waveforms transformed into picocoulombs.
    """
    waveforms_pc = {}

    for var in ConvFactors:
        waveforms_pc[var.name] = l1a_dataset[var.name] * var.value
        waveforms_pc[var.name].attrs = idex_attrs.get_variable_attributes(
            var.name.lower()
        )

    return waveforms_pc
