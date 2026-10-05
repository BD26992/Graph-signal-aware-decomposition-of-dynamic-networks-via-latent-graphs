# Data

This repository includes the small raw and processed datasets used by the
experiments.

## Layout

```text
data/raw/
  surface/
    seasurfacetemperature.mat   Raw SeaSurf signal matrix `R` and coordinates
    surface_hXY.mat             SeaSurf plotting coordinates from the MATLAB code
  UStemp/
    USTemperature.mat           Raw USTemp data from the MATLAB code
    UStemp_hXY.mat              USTemp plotting coordinates
  contact/
    contact.mat                 Raw Contact data from the MATLAB code

data/processed/
  surface_8_data.mat            Paper SeaSurf setting, T=8
  UStemp_15_data.mat            Paper USTemp setting, T=15
  contact_50_data.mat           Paper Contact setting, T=50
```

The processed files are used by the paper reproduction scripts. The raw files
are included so preprocessing choices can be changed, for example rebuilding
SeaSurf with a different number of time slices.

## Rebuilding SeaSurf with a Different `T`

The original paper setting uses `T=8`. To rebuild SeaSurf with another number
of temporal slices, run:

```bash
python3 preprocessing/build_seasurf.py \
  --raw data/raw/surface/seasurfacetemperature.mat \
  --n-times 16 \
  --output data/processed/surface_16_data.mat
```

Increasing `T` gives more temporal slices with fewer graph-signal samples per
slice. Decreasing `T` gives fewer temporal slices with more samples per slice.
