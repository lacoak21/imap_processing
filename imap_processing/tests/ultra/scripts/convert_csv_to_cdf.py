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
    "/Users/luco3133/projects/ultra_stuff/validation_stuff/SES DropBox-bxj69itR5749RG5N"
)
spin_table_name = "SpinTable_flux-2.csv"

kernles = [
    "/Users/luco3133/projects/imap_processing/imap_processing/tests/spice/test_data/imap_sclk_0000.tsc",
    "/Users/luco3133/projects/imap_processing/imap_processing/tests/spice/test_data/naif0012.tls",
]

with sp.KernelPool(kernles) as pool:
    for id in [45, 90]:
        ae_flux = pd.read_csv(f"{folder_path}/AE_IMAP_ULTRA_{id}_flux-2_p0.csv")
        rates_flux = pd.read_csv(f"{folder_path}/Rates_IMAP_ULTRA_{id}_flux-2_p0.csv")
        spin_table = pd.read_csv(f"{folder_path}/SpinTable_flux-2.csv")

        # Create the xarray Dataset with mapped variables
        l1b_ds = xr.Dataset(
            {
                # Time-related variables
                "event_times": (
                    ["epoch"],
                    ae_flux["tdb (s)"].values.astype(np.float32),
                ),
                # Spacecraft position
                # "x_front": (["epoch"], ae_flux["x_inst"].values.astype(np.float32)),
                # "y_front": (["epoch"], ae_flux["y_inst"].values.astype(np.float32)),
                # "x_back": (["epoch"], ae_flux["x_inst"].values.astype(np.float32)),
                # # Assuming same as front for now
                # "y_back": (["epoch"], ae_flux["y_inst"].values.astype(np.float32)),
                # # Assuming same as front for now
                # "x_coin": (["epoch"], ae_flux["x_inst"].values.astype(np.float32)),
                # Assuming same position
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
                "quality_scattering": (
                    ["epoch"],
                    ae_flux["scatter"].values.astype(np.uint16),
                ),
                "quality_outliers": (
                    ["epoch"],
                    ae_flux["accidental"].values.astype(np.uint16),
                ),
                "ebin": (
                    ["epoch"],
                    np.zeros_like(ae_flux["tdb (s)"].values, dtype=np.float64) + 20,
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
