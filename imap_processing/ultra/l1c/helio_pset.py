"""Calculate Pointing Set Grids."""

import logging
import pickle

import astropy_healpix.healpy as hp
import numpy as np
import pandas as pd
import xarray as xr

from imap_processing.cdf.utils import parse_filename_like
from imap_processing.quality_flags import ImapPSETUltraFlags
from imap_processing.spice.time import (
    et_to_met,
    met_to_ttj2000ns,
    ttj2000ns_to_et,
)
from imap_processing.ultra.l1b.ultra_l1b_culling import get_de_rejection_mask
from imap_processing.ultra.l1c.l1c_lookup_utils import (
    build_energy_bins,
    calculate_fwhm_spun_scattering,
    get_spacecraft_pointing_lookup_tables,
)
from imap_processing.ultra.l1c.ultra_l1c_culling import compute_culling_mask
from imap_processing.ultra.l1c.ultra_l1c_pset_bins import (
    get_efficiencies_and_geometric_function,
    get_energy_delta_minus_plus,
    get_helio_adjusted_data,
    get_spacecraft_background_rates,
    get_spacecraft_exposure_times,
    get_spacecraft_histogram,
)
from imap_processing.ultra.utils.ultra_l1_utils import create_dataset

logger = logging.getLogger(__name__)


def calculate_helio_pset(
    de_dataset: xr.Dataset,
    goodtimes_dataset: xr.Dataset,
    rates_dataset: xr.Dataset,
    params_dataset: xr.Dataset,
    name: str,
    ancillary_files: dict,
    instrument_id: int,
    species_id: list,
) -> xr.Dataset | None:
    """
    Create dictionary with defined datatype for Pointing Set Grid Data.

    Parameters
    ----------
    de_dataset : xarray.Dataset
        Dataset containing de data.
    goodtimes_dataset : xarray.Dataset
        Dataset containing goodtimes data.
    rates_dataset : xarray.Dataset
        Dataset containing image rates data.
    params_dataset : xarray.Dataset
        Dataset containing image parameters data.
    name : str
        Name of the dataset.
    ancillary_files : dict
        Ancillary files.
    instrument_id : int
        Instrument ID, either 45 or 90.
    species_id : List
        Species ID.

    Returns
    -------
    dataset : xarray.Dataset
        Dataset containing the data.
    """
    repoint = de_dataset.attrs.get("Repointing", "")
    repoint_id = int(repoint.replace("repoint", ""))

    apply_boundary_scale_factors = False
    sensor_id = int(parse_filename_like(name)["sensor"][0:2])
    pset_dict: dict[str, np.ndarray] = {}
    # Select only the species we are interested in.
    indices = np.where(np.isin(de_dataset["ebin"].values, species_id))[0]
    if indices.size == 0:
        logger.info(f"No data available for {name}")
        return None

    species_dataset = de_dataset.isel(epoch=indices)

    spin_data = pd.read_csv(
        f"/Users/luco3133/projects/ultra_stuff/validation_stuff"
        f"/other_var_validation_20251024/ultra-{sensor_id}-inputs/"
        f"SpinTable-p{repoint_id}.csv"
    )
    pointing_start = et_to_met(float(spin_data["Spin Start (tdb)"].values[0]))
    pointing_mid_time = (
        pointing_start + et_to_met(float(spin_data["Spin Start (tdb)"].values[-1]))
    ) / 2
    pointing_stop = et_to_met(float(spin_data["Spin Start (tdb)"].values[-1]))

    rejected = get_de_rejection_mask(
        species_dataset["quality_scattering"].values,
        species_dataset["quality_outliers"].values,
    )
    species_dataset = species_dataset.isel(epoch=~rejected)

    v_mag_helio_spacecraft = np.linalg.norm(
        species_dataset["velocity_dps_helio"].values, axis=1
    )
    vhat_dps_helio = (
        species_dataset["velocity_dps_helio"].values
        / v_mag_helio_spacecraft[:, np.newaxis]
    )
    intervals, _, energy_bin_geometric_means = build_energy_bins()
    # Get lookup table for FOR indices by spin phase step
    (
        for_indices_by_spin_phase,
        theta_vals,
        phi_vals,
        ra_and_dec,
        boundary_scale_factors,
    ) = get_spacecraft_pointing_lookup_tables(ancillary_files, instrument_id)

    logger.info("calculating spun FWHM scattering values.")
    scattering_file = f"scattering_results_{sensor_id}_helio.pkl"
    try:
        with open(scattering_file, "rb") as f:
            data = pickle.load(f)
        pixels_below_scattering = data["pixels_below_scattering"]
        scattering_theta = data["scattering_theta"]
        scattering_phi = data["scattering_phi"]
        scattering_thresholds = data["scattering_thresholds"]
        logger.info(f"Loaded scattering results from {scattering_file}")
    except (FileNotFoundError, EOFError, pickle.UnpicklingError):
        logger.info("calculating spun FWHM scattering values.")
        pixels_below_scattering, scattering_theta, scattering_phi, scattering_thresholds = (
            calculate_fwhm_spun_scattering(
                for_indices_by_spin_phase,
                theta_vals,
                phi_vals,
                ancillary_files,
                instrument_id,
            )
        )
        # Save all four arrays
        with open(scattering_file, "wb") as f:
            pickle.dump(
                {
                    "pixels_below_scattering": pixels_below_scattering,
                    "scattering_theta": scattering_theta,
                    "scattering_phi": scattering_phi,
                    "scattering_thresholds": scattering_thresholds,
                },
                f,
            )
        logger.info(f"Saved scattering results to {scattering_file}")

    nside = hp.npix2nside(for_indices_by_spin_phase.shape[0])
    counts, latitude, longitude, n_pix = get_spacecraft_histogram(
        vhat_dps_helio,
        species_dataset["energy_heliosphere"].values,
        intervals,
        nside=nside,
    )
    helio_pset_quality_flags = np.full(
        n_pix, ImapPSETUltraFlags.NONE.value, dtype=np.uint16
    )
    healpix = np.arange(n_pix)

    # Get the start and stop times of the pointing period
    repoint_id = species_dataset.attrs.get("Repointing", None)
    if repoint_id is None:
        raise ValueError("Repointing ID attribute is missing from the dataset.")

    # pointing_range_met = get_pointing_times_from_id(repoint_id)
    pointing_range_met = (pointing_start, pointing_stop)
    logger.info("Calculating spacecraft exposure times with deadtime correction.")
    exposure_time, deadtime_ratios = get_spacecraft_exposure_times(
        rates_dataset,
        params_dataset,
        pixels_below_scattering,
        boundary_scale_factors,
        pointing_range_met,
        n_pix=n_pix,
        apply_boundary_scale_factors=apply_boundary_scale_factors,
        sensor_id=sensor_id,
        ancillary_files=ancillary_files,
    )
    logger.info("Calculating spun efficiencies and geometric function.")
    # calculate efficiency and geometric function as a function of energy

    eff_file = f"eff_gf_exp{sensor_id}_helio.pkl"
    try:
        with open(eff_file, "rb") as f:
            data = pickle.load(f)
            deadtime_ratios = data["deadtime_ratios"]
            geometric_function = data["geometric_function"]
            efficiencies = data["efficiencies"]
        logger.info(f"Loaded efficiencies and geometric function from {eff_file}")
    except (FileNotFoundError, EOFError, pickle.UnpicklingError, KeyError):
        logger.info("calculating spun efficiencies and geometric function.")
        geometric_function, efficiencies = get_efficiencies_and_geometric_function(
            pixels_below_scattering,
            boundary_scale_factors,
            theta_vals,
            phi_vals,
            n_pix,
            ancillary_files,
            apply_boundary_scale_factors,
        )
        with open(eff_file, "wb") as f:
            pickle.dump(
                {
                    "deadtime_ratios": deadtime_ratios,
                    "geometric_function": geometric_function,
                    "efficiencies": efficiencies,
                },
                f,
            )
        logger.info(f"Saved efficiencies and geometric function to {eff_file}")
    logger.info("Calculating background rates.")
    # TODO calculate helio background rates
    # Calculate background rates
    background_rates = get_spacecraft_background_rates(
        rates_dataset,
        sensor_id,
        ancillary_files,
        intervals,
        goodtimes_dataset["spin_number"].values,
        nside=nside,
    )

    # mid_time = ttj2000ns_to_et(met_to_ttj2000ns((np.sum(pointing_range_met)) / 2))
    mid_time = ttj2000ns_to_et(met_to_ttj2000ns(pointing_mid_time))

    logger.info("Adjusting data for helio frame.")
    exposure_time, efficiencies, geometric_function = get_helio_adjusted_data(
        mid_time,
        exposure_time,
        geometric_function,
        efficiencies,
        ra_and_dec[:, 0],
        ra_and_dec[:, 1],
        nside=nside,
    )
    sensitivity = efficiencies * geometric_function

    start: float = np.min(species_dataset["event_times"].values)
    end: float = np.max(species_dataset["event_times"].values)

    # Convert pointing start and end time to ttj2000ns
    pointing_range_ns = met_to_ttj2000ns(pointing_range_met)
    # use either the pointing end time + 30 mins or the max event time,
    # whichever is smaller.
    end = min(end + 1800, ttj2000ns_to_et(pointing_range_ns[1]))
    # Time bins in 30 minute intervals
    time_bins = np.arange(start, end, 1800)

    # Compute mask for culling the Earth
    compute_culling_mask(
        time_bins,
        6378.1,  # Earth radius
        helio_pset_quality_flags,
        nside=nside,
    )
    # Epoch should be the start of the pointing
    pset_dict["epoch"] = np.atleast_1d(pointing_range_ns[0]).astype(np.int64)
    pset_dict["epoch_delta"] = np.atleast_1d(np.diff(pointing_range_ns)).astype(
        np.int64
    )
    pset_dict["counts"] = counts[np.newaxis, ...]
    pset_dict["latitude"] = latitude[np.newaxis, ...]
    pset_dict["longitude"] = longitude[np.newaxis, ...]
    pset_dict["energy_bin_geometric_mean"] = energy_bin_geometric_means
    pset_dict["background_rates"] = background_rates[np.newaxis, ...]
    pset_dict["exposure_factor"] = exposure_time[np.newaxis, ...]
    pset_dict["pixel_index"] = healpix
    pset_dict["energy_bin_delta"] = np.diff(intervals, axis=1).squeeze()[
        np.newaxis, ...
    ]
    pset_dict["sensitivity"] = sensitivity
    pset_dict["efficiency"] = efficiencies
    pset_dict["geometric_function"] = geometric_function
    pset_dict["dead_time_ratio"] = deadtime_ratios
    pset_dict["spin_phase_step"] = np.arange(len(deadtime_ratios))
    pset_dict["quality_flags"] = helio_pset_quality_flags[np.newaxis, ...]

    pset_dict["scatter_theta"] = scattering_theta
    pset_dict["scatter_phi"] = scattering_phi
    pset_dict["scatter_threshold"] = scattering_thresholds

    # Add the energy delta plus/minus to the dataset
    energy_delta_minus, energy_delta_plus = get_energy_delta_minus_plus()
    pset_dict["energy_delta_minus"] = energy_delta_minus
    pset_dict["energy_delta_plus"] = energy_delta_plus

    dataset = create_dataset(pset_dict, name, "l1c")
    dataset.attrs["Repointing"] = repoint
    return dataset
