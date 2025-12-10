#!/bin/bash


helio=true

# ...do something interesting...
if [ "$helio" = true ]
  then
  descriptor="90sensor-heliopset"
  spice_files='{"type": "spice","files": ["imap_pred_20250401_20250501_v01.bsp","imap_pred_od005_20251014_20251125_v01.bsp","de440.bsp", "imap_dps_2026_268_2026_268_01.ah.bc", "imap_dps_2026_269_2026_269_01.ah.bc", "imap_science_105.tf", "imap_science_100.tf", "naif016.tls", "imap_sclk_005.tsc", "imap_2026_269_2026_269_01.ah.bc", "imap_2026_268_2026_268_01.ah.bc", "imap_dps_2025_105_2026_105_009.ah.bc", "imap_2025_105_2026_105_01.ah.bc"]}'
  de_version="v001"
  else
  descriptor="90sensor-spacecraftpset"
  spice_files='{"type": "spice","files": ["imap_pred_od005_20251014_20251125_v01.bsp","imap_recon_20250414_20260415_v01.bsp","imap_pred_od005_20251014_20251125_v01.bsp","de440.bsp","imap_recon_20250415_20260415_v01.bsp", "imap_dps_2026_268_2026_268_01.ah.bc", "imap_dps_2026_269_2026_269_01.ah.bc", "imap_science_105.tf", "imap_science_100.tf", "naif016.tls", "imap_sclk_005.tsc", "imap_2026_269_2026_269_01.ah.bc", "imap_2026_268_2026_268_01.ah.bc", "imap_dps_2025_105_2026_105_009.ah.bc", "imap_2025_105_2026_105_01.ah.bc"]}'
  de_version="v000"
fi

export IMAP_DATA_ACCESS_URL=https://api.dev.imap-mission.com

# STEP 1: RUN FIRST PSET WITH COMMAND:
# imap_cli --instrument ultra --data-level l1c --descriptor 90sensor-spacecraftpset --start-date 20260926 --version v001 --dependency imap_ultra_l1c_90sensor-spacecraftpset-564af5aa_20251011-repoint00001_v001.json
# imap_cli --instrument ultra --data-level l1c --descriptor 45sensor-spacecraftpset --start-date 20260926 --version v001 --dependency imap_ultra_l1c_45sensor-spacecraftpset-564af5aa_20251011-repoint00001_v001.json
# Fixed repoint file (same for all)
repoint_file="imap_2026_269_15.repoint"

# Fixed spin table name (same for all, gets overwritten each time)
spin_table="imap_2026_268_2026_269_01.spin.csv"

# Track total time
script_start=$(date +%s)

# P0 should already have been run so we can copy eff and gf and spun scattering ect..
# Loop through pointings 1 to 184
for pointing in {0..184}; do
    # Start timing for this pointing
    start_time=$(date +%s)

    # Pad pointing to 5 digits
    pointing_padded=$(printf "%05d" "$pointing")

    # Set the de and rates filenames with repointing
    de_file="imap_ultra_l1b_90sensor-de_20000101-repoint${pointing_padded}_${de_version}.cdf"
    rates_file="imap_ultra_l1a_90sensor-rates_20000101-repoint${pointing_padded}_v000.cdf"

    echo "Processing pointing ${pointing}"

    # Run the command
    imap_cli --instrument ultra \
        --data-level l1c \
        --descriptor $descriptor \
        --start-date $(date -j -v+${pointing}d -f "%Y%m%d" "20250416" +"%Y%m%d") \
        --version v000 \
        --dependency '[
            {"type": "science","files": ["imap_ultra_l1b_90sensor-goodtimes_20260926-repoint02003_v002.cdf"]},
            {"type": "science","files": ["'"$de_file"'"]},
            {"type": "science","files": ["'"$rates_file"'"]},
            {"type": "science","files": ["imap_ultra_l1a_90sensor-params_20260926-repoint02003_v001.cdf"]},
            {"type": "ancillary","files": ["imap_ultra_l1c-90sensor-sc-pointing-bsf_20250101_v001.csv"]},
            {"type": "ancillary","files": ["imap_ultra_l1c-90sensor-sc-pointing-theta_20250101_v001.csv"]},
            {"type": "ancillary","files": ["imap_ultra_l1c-90sensor-sc-pointing-phi_20250101_v001.csv"]},
            {"type": "ancillary","files": ["imap_ultra_l1c-90sensor-sc-pointing-index_20250101_v001.csv"]},
            {"type": "ancillary","files": ["imap_ultra_l1b-90sensor-scattering-calibration-data_20250101_v000.csv"]},
            {"type": "ancillary","files": ["imap_ultra_l1b-scattering-thresholds-per-energy_20250101_v001.csv"]},
            {"type": "ancillary","files": ["imap_ultra_l1b-45sensor-logistic-interpolation_20250101_v000.csv"]},
            {"type": "ancillary","files": ["imap_ultra_l1b-90sensor-imgparams-lookup_20250101_v001.csv"]},
            {"type": "ancillary","files": ["imap_ultra_l1b-sensor-gf-blades_20250101_v000.csv"]},
            {"type": "repoint","files": ["'"$repoint_file"'"]},
            {"type": "spin","files": ["imap_2026_268_2026_269_01.spin.csv"]},
            '"$spice_files"'
        ]'
  # space craft  ["imap_pred_od005_20251014_20251125_v01.bsp","imap_recon_20250414_20260415_v01.bsp","imap_pred_od005_20251014_20251125_v01.bsp","de440.bsp","imap_recon_20250415_20260415_v01.bsp", "imap_dps_2026_268_2026_268_01.ah.bc", "imap_dps_2026_269_2026_269_01.ah.bc", "imap_science_105.tf", "imap_science_100.tf", "naif016.tls", "imap_sclk_005.tsc", "imap_2026_269_2026_269_01.ah.bc", "imap_2026_268_2026_268_01.ah.bc", "imap_dps_2025_105_2026_105_009.ah.bc", "imap_2025_105_2026_105_01.ah.bc"]}
    # helio             #{"type": "spice","files": ["imap_pred_20250401_20250501_v01.bsp","imap_pred_od005_20251014_20251125_v01.bsp","de440.bsp", "imap_dps_2026_268_2026_268_01.ah.bc", "imap_dps_2026_269_2026_269_01.ah.bc", "imap_science_105.tf", "imap_science_100.tf", "naif016.tls", "imap_sclk_005.tsc", "imap_2026_269_2026_269_01.ah.bc", "imap_2026_268_2026_268_01.ah.bc", "imap_dps_2025_105_2026_105_009.ah.bc", "imap_2025_105_2026_105_01.ah.bc"]}
    # Calculate elapsed time
    end_time=$(date +%s)
    elapsed=$((end_time - start_time))

    # Convert to minutes and seconds
    minutes=$((elapsed / 60))
    seconds=$((elapsed % 60))

    echo "Completed pointing ${pointing} in ${minutes}m ${seconds}s"
    echo "---"
done

# Calculate total time
script_end=$(date +%s)
total_elapsed=$((script_end - script_start))
total_minutes=$((total_elapsed / 60))
total_seconds=$((total_elapsed % 60))

echo "All 185 pointings processed!"
echo "Total time: ${total_minutes}m ${total_seconds}s"

# IMAP_DATA_ACCESS_URL=https://api.dev.imap-mission.com
# imap_cli --instrument ultra --data-level l2 --descriptor u45-ena-h-sc-nsp-full-hae-2deg-3mo --start-date 20250415 --version v001 --dependency imap_ultra_l2_u45-ena-h-sc-nsp-full-hae-2deg-3mo-564af5aa_20251011_v001.json
# imap_cli --instrument ultra --data-level l2 --descriptor u90-ena-h-sc-nsp-full-hae-2deg-3mo --start-date 20250415 --version v001 --dependency imap_ultra_l2_u90-ena-h-sc-nsp-full-hae-2deg-3mo-564af5aa_20251011_v001.json
# imap_cli --instrument ultra --data-level l2 --descriptor u45-ena-h-hf-nsp-full-hae-2deg-3mo --start-date 20250415 --version v001 --dependency imap_ultra_l2_u45-ena-h-hf-nsp-full-hae-2deg-3mo-564af5aa_20251011_v001.json
# imap_cli --instrument ultra --data-level l2 --descriptor u90-ena-h-hf-nsp-full-hae-2deg-3mo --start-date 20250415 --version v001 --dependency imap_ultra_l2_u90-ena-h-hf-nsp-full-hae-2deg-3mo-564af5aa_20251011_v001.json

# healpix
# imap_cli --instrument ultra --data-level l2 --descriptor u45-ena-h-hf-nsp-full-hae-nside32-3mo --start-date 20250415 --version v001 --dependency imap_ultra_l2_u45-ena-h-hf-nsp-full-hae-2deg-3mo-564af5aa_20251011_v001.json
# imap_cli --instrument ultra --data-level l2 --descriptor u90-ena-h-hf-nsp-full-hae-nside32-3mo --start-date 20250415 --version v001 --dependency imap_ultra_l2_u90-ena-h-hf-nsp-full-hae-2deg-3mo-564af5aa_20251011_v001.json
# imap_cli --instrument ultra --data-level l2 --descriptor u45-ena-h-sc-nsp-full-hae-nside32-3mo --start-date 20250415 --version v001 --dependency imap_ultra_l2_u45-ena-h-sc-nsp-full-hae-2deg-3mo-564af5aa_20251011_v001.json
# imap_cli --instrument ultra --data-level l2 --descriptor u90-ena-h-sc-nsp-full-hae-nside32-3mo --start-date 20250415 --version v001 --dependency imap_ultra_l2_u90-ena-h-sc-nsp-full-hae-2deg-3mo-564af5aa_20251011_v001.json