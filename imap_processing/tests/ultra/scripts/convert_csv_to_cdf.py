# %%
import numpy as np
import pandas as pd
import xaray as xr

folder_path = (
    "/Users/luco3133/projects/ultra_stuff/validation_stuff/SES DropBox-bxj69itR5749RG5N"
)
spin_table_name = "SpinTable_flux-2.csv"

for id in [45, 90]:
    ae_flux = pd.read_csv(f"{folder_path}/AE_IMAP_ULTRA_{id}_flux-2_p0.csv")
    rates_flux = pd.read_csv(f"{folder_path}/Rates_IMAP_ULTRA_{id}_flux-2_p0.csv")
    spin_table = pd.read_csv(f"{folder_path}/SpinTable_flux-2.csv")

    # Create epoch dimension (assuming each row is a time step/epoch)
    n_epochs = len(ae_flux)

    # Create the xarray Dataset with mapped variables
    ds = xr.Dataset(
        {
            # Time-related variables
            "event_times": (["epoch"], ae_flux["tdb (s)"].values.astype(np.float32)),
            # Spacecraft position
            "x_front": (["epoch"], ae_flux["x_inst"].values.astype(np.float32)),
            "y_front": (["epoch"], ae_flux["y_inst"].values.astype(np.float32)),
            "x_back": (["epoch"], ae_flux["x_inst"].values.astype(np.float32)),
            # Assuming same as front for now
            "y_back": (["epoch"], ae_flux["y_inst"].values.astype(np.float32)),
            # Assuming same as front for now
            "x_coin": (["epoch"], ae_flux["x_inst"].values.astype(np.float32)),
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
            "event_efficiency": (["epoch"], ae_flux["eff"].values.astype(np.float64)),
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
            # Mapping accidental to outliers
        },
        coords={
            "epoch": ae_flux["tdb (s)"].values.astype(np.float64),
            "component": ["x", "y", "z"],  # For 3D vector components
        },
    )

    # print(ae_flux.columns)
    # print(rates_flux.columns)
    # print(spin_table.columns)
    # print(len(ae_flux), len(rates_flux), len(spin_table))

# # %%
# from imap_processing.cdf.utils import load_cdf
#
# ds = load_cdf(
#     "/Users/luco3133/Downloads/imap_ultra_l1b_45sensor-
#     de_20240207-repoint99999_v999.cdf"
# )
# ds.data_vars
