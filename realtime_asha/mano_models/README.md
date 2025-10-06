# MANO Models Directory

This directory should contain the MANO hand model files.

## Required Files

After conversion, you should have:
- `MANO_RIGHT_CONVERTED.pkl`
- `MANO_LEFT_CONVERTED.pkl`

## How to Get MANO Models

1. Visit https://mano.is.tue.mpg.de/download.php
2. Register and download MANO v1.2
3. Extract `MANO_RIGHT.pkl` and `MANO_LEFT.pkl`
4. Place them in this directory
5. Run the conversion script:
   ```bash
   source mano_convert_env/bin/activate
   python ../src/convert_mano_to_numpy.py
   ```

## Why Are Models Not Included?

MANO models require separate licensing and cannot be redistributed.
You must download them directly from the MANO website.

## File Structure After Setup

```
mano_models/
├── README.md                     # This file
├── MANO_RIGHT.pkl                # Original (backup)
├── MANO_LEFT.pkl                 # Original (backup)
├── MANO_RIGHT_CONVERTED.pkl      # Converted (used by system)
└── MANO_LEFT_CONVERTED.pkl       # Converted (used by system)
```

**Note**: Only `README.md` is tracked by git. Model files are excluded via `.gitignore`.
