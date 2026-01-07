# Motion Quality Metrics

Companion code for: **"Toward a Cross-Domain Taxonomy of Motion Quality Metrics"** (FG2026)


![The conceptual level with the four quality categories and their corresponding operational levels with indicative illustrations
](images/metrics_figure.png)


## Overview

This repository implements a comprehensive framework for computing motion quality metrics from BVH motion capture data, organized into four motion quality families:

1. **Fluency** - Smoothness, coordination, and continuous flow
2. **Balance & Symmetry** - Postural stability and bilateral coordination
3. **Spatial Energy** - Movement amplitude and dynamic intensity
4. **Variability & Consistency** - Reproducibility across contexts

## Project Structure

```
motion_quality_metrics/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── run_analysis.py             # Main analysis script
├── data_loader.py              # BVH data loading and preprocessing
├── fluency.py                  # Fluency metrics (8 functions)
├── balance_symmetry.py         # Balance & Symmetry metrics (10 functions)
├── spatial_energy.py           # Spatial Energy metrics (10 functions)
├── variability.py              # Variability metrics (5 functions)
├── pymo/                       # PyMO library for BVH parsing
└── bvh2/                       # Sample BVH motion capture files
```

## Installation

1. **Clone or download this repository**

2. **Install required Python packages:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python -c "import numpy, pandas, scipy, sklearn, matplotlib; print('All dependencies installed!')"
   ```

## Quick Start

Run the complete analysis pipeline:

```bash
python run_analysis.py
```

This will:

- Load a sample BVH file from `bvh2/`
- Compute all motion quality metrics
- Display results with interpretations

## Usage Examples

### Using Individual Modules

```python
from data_loader import load_bvh, calculate_com
from fluency import log_dimensionless_jerk, spectral_arc_length
from balance_symmetry import sway_amplitude_rms
from spatial_energy import quantity_of_movement
from variability import body_posture_variation

# Load BVH data
parsed_data, mocap_data, frame_rate, joints = load_bvh('bvh2/MCEAS02G01R03.bvh')

# Compute metrics
ldj = log_dimensionless_jerk(trajectory, frame_rate)
sparc = spectral_arc_length(trajectory, frame_rate)
sway = sway_amplitude_rms(com_trajectory)
```

### Custom Analysis Pipeline

See `run_analysis.py` for a complete example of:

- Loading BVH files
- Extracting rotations and positions
- Computing center of mass
- Calculating all metric families

## Metrics Overview

### Fluency Metrics (fluency.py)

- **Temporal Smoothness**: Log Dimensionless Jerk (LDLJ), Spectral Arc Length (SPARC), Local Maxima count
- **Postural Continuity**: Body Posture Rate (BPR)
- **Spatial Regularity**: Curvature Variability, Tortuosity
- **Rhythmic Consistency**: Harmonic Ratio, Velocity Autocorrelation Decay

### Balance & Symmetry (balance_symmetry.py)

- **Static Balance**: Sway Amplitude RMS, Mean Velocity of Sway, Sway Area
- **Dynamic Balance**: XCoM, Margin of Stability, Local Dynamic Stability (Lyapunov)
- **Spatiotemporal Symmetry**: Gait Symmetry Index (GSI)
- **Kinematic Symmetry**: Bilateral Trajectory RMSE, Correlation, GQoM

### Spatial Energy (spatial_energy.py)

- **Amplitude**: Joint Range of Motion, Reach Extensiveness, Gesture Volume, Convex Hull
- **Energy/Intensity**: Quantity of Movement, Weighted QoM, Kinetic Energy proxy
- **Efficiency**: Path Directness Index, Trajectory Curvature, Energy Economy

### Variability & Consistency (variability.py)

- **Intra-subject, Intra-motion**: Body Posture Variation, Trajectory Variability, Trial-to-trial SD
- **Intra-subject, Inter-motion**: Consistency SD
- **Inter-subject**: Coefficient of Variability (CoV)

## Data Format

The code expects BVH (BioVision Hierarchy) motion capture files. Sample files are provided in the `bvh2/` directory.

## Documentation

Each module contains comprehensive in-code documentation with:

- Parameter descriptions
- Return value specifications
- Mathematical formulas
- Academic references
- Usage examples


## Citation

```
[Citation information for FG2026 paper - to be added upon publication]
```

## Requirements

- Python 3.8+
- NumPy >= 1.24.0
- Pandas >= 2.0.0
- SciPy >= 1.10.0
- scikit-learn >= 1.3.0
- Matplotlib >= 3.7.0
- PyMO (included in `pymo/` directory)

## License

[License information to be added]

## Authors

[Author information to be added]

## Acknowledgments

- PyMO library for BVH parsing: https://github.com/omimo/PyMO
- Sample BVH data from [data source to be credited]]
