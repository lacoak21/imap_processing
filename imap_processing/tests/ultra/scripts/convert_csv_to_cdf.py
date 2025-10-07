# %%
import numpy as np
import pandas as pd
import spiceypy as sp
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.utils import write_cdf

cdf_manager = ImapCdfAttributes()
cdf_manager.add_instrument_global_attrs("ultra")
cdf_manager.add_instrument_variable_attrs("ultra", "l1b")

folder_path = (
    "/Users/luco3133/projects/ultra_stuff/validation_stuff/flux_2_basic_corrected"
)
spin_table_name = "SpinTable-flux-2-basic-noeff.csv"

kernles = [
    "/Users/luco3133/projects/imap_processing/imap_processing/tests/spice/test_data/imap_sclk_0000.tsc",
    "/Users/luco3133/projects/imap_processing/imap_processing/tests/spice/test_data/naif0012.tls",
]

with sp.KernelPool(kernles) as pool:
    for id in [90]:
        ae_flux = pd.read_csv(
            f"{folder_path}/AE-IMAP_ULTRA_{id}-flux-2-basic-noeff-p0.csv"
        )
        # print(ae_flux)
        rates_flux = pd.read_csv(
            f"{folder_path}/Rates-IMAP_ULTRA_{id}-flux-2-basic-noeff-p0.csv"
        )
        # spin_table = pd.read_csv(f"{folder_path}/SpinTable_flux-2.csv")
        print(f"min energy {id}:", np.min(np.array(ae_flux["energy_sc"].values)))
        print(f"max energy {id}:", np.max(np.array(ae_flux["energy_sc"].values)))
        # print(spin_table)
        energy_max = 1000
        # Create the xarray Dataset with mapped variables
        ebin = np.where(
            (ae_flux["accidental"].values & (ae_flux["energy_sc"] > energy_max)), 255, 1
        )
        print(ebin)
        l1b_ds = xr.Dataset(
            {
                # Time-related variables
                "event_times": (
                    ["epoch"],
                    ae_flux["tdb (s)"].values.astype(np.float32),
                ),
                # Spacecraft velocity components
                "velocity_sc": (
                    ["epoch", "component"],
                    np.stack(
                        [
                            ae_flux["v_x_sc"].values,
                            ae_flux["v_y_sc"].values,
                            ae_flux["v_z_sc"].values,
                        ],
                        axis=1,
                    ).astype(np.float32),
                ),
                "velocity_dps_sc": (
                    ["epoch", "component"],
                    np.stack(
                        [
                            ae_flux["v_x_sc"].values,
                            ae_flux["v_y_sc"].values,
                            ae_flux["v_z_sc"].values,
                        ],
                        axis=1,
                    ).astype(np.float32),
                ),
                # Heliospheric velocity components
                "velocity_dps_helio": (
                    ["epoch", "component"],
                    np.stack(
                        [
                            ae_flux["v_x_hel"].values,
                            ae_flux["v_y_hel"].values,
                            ae_flux["v_z_hel"].values,
                        ],
                        axis=1,
                    ).astype(np.float32),
                ),
                # Energy variables
                "energy_spacecraft": (
                    ["epoch"],
                    ae_flux["energy_sc"].values.astype(np.float32),
                ),
                "energy_heliosphere": (
                    ["epoch"],
                    ae_flux["energy_hel"].values.astype(np.float32),
                ),
                "energy": (["epoch"], ae_flux["energy_sc"].values.astype(np.float32)),
                # Primary energy variable
                # Event efficiency and geometric factor
                "event_efficiency": (
                    ["epoch"],
                    ae_flux["eff"].values.astype(np.float64),
                ),
                "geometric_factor_blades": (
                    ["epoch"],
                    ae_flux["gf"].values.astype(np.float64),
                ),
                # Quality flags
                # "quality_scattering": (
                #     ["epoch"],
                #     ae_flux["scatter"].values.astype(np.uint16),
                # ),
                # "quality_outliers": (
                #     ["epoch"],
                #     ae_flux["accidental"].values.astype(np.uint16),
                # ),
                "ebin": (
                    ["epoch"],
                    ebin,
                ),
                # Mapping accidental to outliers
            },
            coords={
                "epoch": ae_flux["tdb (s)"].values.astype(np.float64),
                "component": ["x", "y", "z"],  # For 3D vector components
            },
            attrs=cdf_manager.get_global_attributes(f"imap_ultra_l1b_{id}sensor-de"),
        )
        rates_ds = xr.Dataset(
            {
                # Spacecraft velocity components
                "spin_phase": (
                    ["epoch"],
                    rates_flux["Spin Phase (deg)"].values.astype(np.float32),
                ),
                "start_rate": (
                    ["epoch"],
                    rates_flux["Start Rate (Hz)"].values.astype(np.float32),
                ),
                "stop_rate": (
                    ["epoch"],
                    rates_flux["Stop Rate (Hz)"].values.astype(np.float32),
                ),
                "coin_rate": (
                    ["epoch"],
                    rates_flux["Coin Rate (Hz)"].values.astype(np.float32),
                ),
                "dead_time_ratio": (
                    ["epoch"],
                    rates_flux["Dead Time Ratio"].values.astype(np.float32),
                ),
            },
            coords={
                "epoch": np.arange(len(rates_flux["Dead Time Ratio"].values)),
            },
            attrs=cdf_manager.get_global_attributes(f"imap_ultra_l1a_{id}sensor-rates"),
        )
        l1b_ds.attrs["Data_version"] = "100"
        rates_ds.attrs["Data_version"] = "100"
        write_cdf(l1b_ds)
        write_cdf(rates_ds)

# %%
# from imap_processing.cdf.utils import load_cdf, write_cdf
#
# ds = load_cdf(
#     "/Users/luco3133/Downloads/
#     imap_ultra_l1b_45sensor-de_20240207-repoint99999_v999.cdf"
# )
# print(ds.ebin.data)

# ds2 = load_cdf(
#     "/Users/luco3133/projects/imap_processing/
#     data/imap/ultra/l1a/2026/09/imap_ultra_l1a_90sensor-rates_20260926_v006.cdf"
# )
# ds2.data_vars

"""
imap_cli --instrument ultra --data-level l1c --descriptor 90sensor-spacecraftpset
--start-date 20260926 --version v001 --dependency '[{"type": "science","files":
 ["imap_ultra_l1b_90sensor-goodtimes_20260926_v002.cdf"]},{"type": "science","files":
  ["imap_ultra_l1b_90sensor-de_20000101_v999.cdf"]},{"type": "science","files":
   ["imap_ultra_l1a_90sensor-rates_20000101_v999.cdf"]},{"type": "science","files":
   ["imap_ultra_l1a_90sensor-params_20260926_v006.cdf"]},{"type": "ancillary","files":
   ["imap_ultra_l1c-90sensor-sc-pointing-bsf_20250101_v000.csv"]},{"type": "ancillary",
   "files": ["imap_ultra_l1c-90sensor-sc-pointing-theta_20250101_v000.csv"]},{"type":
   "ancillary","files": ["imap_ultra_l1c-90sensor-sc-pointing-phi_20250101_v000.csv"]},
   {"type": "ancillary","files":
   ["imap_ultra_l1c-90sensor-sc-pointing-index_20250101_v000.csv"]},
   {"type": "ancillary","files":
   ["imap_ultra_l1b-90sensor-scattering-calibration-data_20250101_v000.csv"]},
   {"type": "ancillary","files":
    ["imap_ultra_l1b-scattering-thresholds-per-energy_20250101_v000.csv"]},
    {"type": "ancillary","files":
     ["imap_ultra_l1b-45sensor-logistic-interpolation_20250101_v000.csv"]},
     {"type": "ancillary","files":
     ["imap_ultra_l1b-90sensor-imgparams-lookup_20250101_v001.csv"]},
     {"type": "ancillary","files":
     ["imap_ultra_l1b-sensor-gf-blades_20250101_v000.csv"]},
     {"type": "repoint","files": ["imap_2026_269_05.repoint.csv"]},
     {"type": "spin","files": ["imap_2026_268_2026_269_01.spin.csv"]},
     {"type": "spice", "files":
     ["imap_recon_20250415_20260415_v01.bsp", "imap_dps_2026_268_2026_268_01.ah.bc",
      "imap_science_105.tf", "imap_science_100.tf", "naif016.tls",
       "imap_sclk_005.tsc","imap_2026_269_2026_269_10.ah.bc"]}]'
       """
