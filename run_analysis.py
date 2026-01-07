"""
Motion Quality Metrics - Main Analysis Script

Companion code for: "Toward a Cross-Domain Taxonomy of Motion Quality Metrics" (FG2026)

This script demonstrates the complete workflow for computing motion quality metrics
from BVH motion capture data, following the four motion quality metrics:


FOUR MOTION QUALITY FAMILIES:
==============================

1) FLUENCY - Smoothness, coordination, and continuous flow of movement
   Conceptual: How jerky vs. flowing is the movement?
   Operational definitions:
   - Temporal Smoothness: Continuity of derivatives
     Computational metrics: Log Dimensionless Jerk (LDLJ), Spectral Arc Length (SPARC), Local Maxima count
   - Postural Continuity: Whole-body configuration changes
     Computational metrics: Body Posture Rate (BPR)
   - Spatial Regularity: Path predictability and curvature consistency
     Computational metrics: Curvature Variability, Tortuosity
   - Rhythmic Consistency: Timing predictability in cyclic movements
     Computational metrics: Harmonic Ratio, Velocity Autocorrelation Decay (VAD)

2) BALANCE & SYMMETRY - Postural stability and bilateral coordination
   Conceptual: How stable and symmetric is the movement?
   Operational definitions:
   - Static Balance: CoM control during stationary stance
     Computational metrics: Sway Amplitude RMS, Mean Velocity of Sway, Sway Area (95% confidence ellipse)
   - Dynamic Balance: Stability during movement and perturbations
     Computational metrics: Extrapolated Center of Mass (XCoM), Margin of Stability (MoS), Local Dynamic Stability (Lyapunov Exponent)
   - Spatiotemporal Symmetry: Timing balance between left/right
     Computational metrics: Gait Symmetry Index (GSI)
   - Kinematic Symmetry: Trajectory similarity between limbs
     Computational metrics: Bilateral Trajectory RMSE, Correlation of Joint Trajectories, Generalized Quantity of Movement (GQoM)

3) SPATIAL ENERGY - Movement amplitude and dynamic intensity
   Conceptual: How expansive and vigorous is the movement?
   Operational definitions:
   - Amplitude/Expansiveness: Workspace utilization and reach
     Computational metrics: Joint Range of Motion (ROM), Reach Extensiveness Index, Gesture Volume, Convex Hull Volume
   - Energy/Intensity: Movement vigor and workload
     Computational metrics: Quantity of Movement (QoM), Weighted QoM, Kinetic Energy proxy
   - Efficiency/Directness: Economy of movement paths
     Computational metrics: Path Directness Index, Trajectory Curvature Index, Energy Economy

4) VARIABILITY & CONSISTENCY - Reproducibility across contexts
   Conceptual: How consistent is the movement quality?
   Operational definitions:
   - Intra-subject, Intra-motion: Trial-to-trial consistency
     Computational metrics: Body Posture Variation (BPV), Trajectory Variability, Trial-to-trial Standard Deviation
   - Intra-subject, Inter-motion: Consistency across different movements
     Computational metrics: Consistency SD across gestures/conditions
   - Inter-subject, Intra-motion: Consistency across individuals
     Computational metrics: Coefficient of Variability (CoV)


DATA PROCESSING PIPELINE:
=========================

Section 1: Load BVH file
   → Parse hierarchical skeleton structure
   → Extract motion channels (rotations, positions)
   → Determine frame rate and timing

Section 2: Extract Local Rotations
   → Joint angles in local coordinate frames
   → Euler angle representation (XYZ order)
   → Used for: Fluency metrics (angular smoothness)

Section 3: Compute Global Positions
   → Forward kinematics to world coordinates
   → 3D positions for all joints at all frames
   → Used for: Balance, Symmetry, Spatial Energy metrics

Section 4: Compute Center of Mass
   → Winter's anthropometric model
   → Weighted sum of body segment positions
   → Used for: Balance metrics (sway, stability)

Section 5: Calculate Metrics
   → Import metric functions from specialized modules
   → Compute metrics across all four families
   → Display results with interpretations


USAGE:
======

Run this script directly:
    python run_analysis.py

Or import modules for custom analysis:
    from data_loader import load_bvh, calculate_com
    from fluency import log_dimensionless_jerk
    from balance_symmetry import sway_amplitude_rms


For specific metric references, see docstrings in individual metric modules:
- fluency.py
- balance_symmetry.py
- spatial_energy.py
- variability.py
"""

import numpy as np
import matplotlib.pyplot as plt

# Import data loading functions
from data_loader import (
    load_bvh,
    extract_rotations,
    compute_global_positions,
    calculate_com
)

# Import metric functions
from fluency import (
    log_dimensionless_jerk,
    spectral_arc_length,
    local_maxima_count,
    body_posture_rate,
    curvature_variability,
    tortuosity,
    harmonic_ratio,
    velocity_autocorrelation_decay
)

from balance_symmetry import (
    sway_amplitude_rms,
    mean_velocity_of_sway,
    sway_area_ellipse,
    extrapolated_center_of_mass,
    margin_of_stability,
    local_dynamic_stability,
    gait_symmetry_index,
    bilateral_trajectory_rmse,
    correlation_joint_trajectories,
    generalized_quantity_of_movement
)

from spatial_energy import (
    gesture_volume,
    joint_range_of_motion,
    reach_extensiveness_index,
    convex_hull_volume,
    quantity_of_movement,
    weighted_qom,
    kinetic_energy_proxy,
    path_directness_index,
    trajectory_curvature_index,
    energy_economy
)

from variability import (
    body_posture_variation,
    trajectory_variability,
    trial_to_trial_sd,
    consistency_sd,
    coefficient_of_variability
)


# Configure plotting
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10


def main():
    """
    Main analysis workflow: Load data → Compute metrics → Display results
    """
    
    print("="*80)
    print("MOTION QUALITY METRICS ANALYSIS")
    print("Companion code for: 'Toward a Cross-Domain Taxonomy' (FG2026)")
    print("="*80)
    
    # ========================================================================
    # SECTION 1-4: DATA LOADING AND PREPROCESSING
    # ========================================================================
    
    print("\n" + "="*80)
    print("DATA LOADING AND PREPROCESSING")
    print("="*80)
    
    # Load BVH file
    bvh_file = 'bvh2/MCEAS02G01R03.bvh'
    parsed_data, mocap_data, frame_rate, unique_joints = load_bvh(bvh_file)
    
    # Extract local rotations
    local_rotations = extract_rotations(mocap_data)
    
    # Compute global positions
    global_positions = compute_global_positions(parsed_data)
    
    # Calculate Center of Mass
    com_trajectory = calculate_com(global_positions)
    
    # ========================================================================
    # SECTION 5.1: FLUENCY METRICS
    # ========================================================================
    
    print("\n" + "="*80)
    print("5.1 FLUENCY METRICS - Smoothness & Coordination")
    print("="*80)
    
    # Extract right hand trajectory for testing
    right_hand_pos = global_positions[['RightHand_Xposition', 'RightHand_Yposition',
                                        'RightHand_Zposition']].values
    
    # 5.1.1 Temporal Smoothness
    # 5.1.2 Postural Continuity
    print("\n--- 5.1.2 Postural Continuity ---")
    
    # Reshape positions for body_posture_rate: (n_frames, n_joints, 3)
    # global_positions has columns like Joint_Xposition, Joint_Yposition, Joint_Zposition
    # So we have n_joints * 3 columns total
    n_frames = len(global_positions)
    n_position_joints = len(global_positions.columns) // 3  # Divide by 3 for X,Y,Z
    joint_positions = global_positions.values.reshape(n_frames, n_position_joints, 3)
    
    bpr = body_posture_rate(joint_positions, frame_rate=frame_rate)
    print(f"Body Posture Rate: {bpr:.4f} units/sec")
    print(f"  → Lower = smoother postural transitions")
    
    print("\n--- 5.1.1 Temporal Smoothness ---")
    
    ldj = log_dimensionless_jerk(right_hand_pos, frame_rate=frame_rate)
    print(f"Log Dimensionless Jerk: {ldj:.4f}")
    print(f"  → More negative = smoother movement")
    
    sparc = spectral_arc_length(right_hand_pos, frame_rate=frame_rate)
    print(f"\nSpectral Arc Length (SPARC): {sparc:.4f}")
    print(f"  → More negative = smoother movement")
    
    n_maxima = local_maxima_count(right_hand_pos, frame_rate=frame_rate)
    print(f"\nLocal Maxima Count: {n_maxima}")
    print(f"  → Fewer peaks = smoother movement")
    
    # 5.1.3 Spatial Regularity
    print("\n--- 5.1.3 Spatial Regularity ---")
    
    curv_var = curvature_variability(right_hand_pos, frame_rate=frame_rate)
    print(f"Curvature Variability: {curv_var:.6f}")
    print(f"  → Lower = more regular path curvature")
    
    tort = tortuosity(right_hand_pos)
    print(f"\nTortuosity: {tort:.4f}")
    print(f"  → Closer to 1.0 = more direct path")
    
    # 5.1.4 Rhythmic Consistency
    print("\n--- 5.1.4 Rhythmic Consistency ---")
    
    hr = harmonic_ratio(right_hand_pos, frame_rate=frame_rate)
    print(f"Harmonic Ratio: {hr:.4f}")
    print(f"  → Higher = more rhythmic movement")
    
    vad = velocity_autocorrelation_decay(right_hand_pos, frame_rate=frame_rate)
    print(f"\nVelocity Autocorrelation Decay: {vad:.4f} sec")
    print(f"  → Lower = faster decorrelation (less rhythmic)")
    
    # ========================================================================
    # SECTION 5.2: BALANCE & SYMMETRY METRICS
    # ========================================================================
    
    print("\n" + "="*80)
    print("5.2 BALANCE & SYMMETRY METRICS - Stability & Coordination")
    print("="*80)
    
    # 5.2.1 Static Balance
    print("\n--- 5.2.1 Static Balance ---")
    
    rms_sway = sway_amplitude_rms(com_trajectory, axis='both')
    print(f"Sway Amplitude RMS: {rms_sway:.2f} spatial units")
    print(f"  → Lower = better postural control")
    
    vel_sway = mean_velocity_of_sway(com_trajectory, frame_rate=frame_rate, axis='both')
    print(f"\nMean Velocity of Sway: {vel_sway:.2f} units/sec")
    print(f"  → Lower = less postural corrections")
    
    area_sway = sway_area_ellipse(com_trajectory, confidence=0.95)
    print(f"\nSway Area (95% ellipse): {area_sway:.2f} units²")
    print(f"  → Smaller = better balance control")
    
    # 5.2.2 Dynamic Balance
    print("\n--- 5.2.2 Dynamic Balance ---")
    
    lambda_max = local_dynamic_stability(com_trajectory, frame_rate=frame_rate)
    print(f"Local Dynamic Stability (Lyapunov): {lambda_max:.4f} bits/sec")
    print(f"  → Positive values indicate local instability")
    
    # 5.2.4 Kinematic Symmetry
    print("\n--- 5.2.4 Kinematic Symmetry ---")
    
    left_hand = global_positions[['LeftHand_Xposition', 'LeftHand_Yposition',
                                   'LeftHand_Zposition']].values
    right_hand = global_positions[['RightHand_Xposition', 'RightHand_Yposition',
                                    'RightHand_Zposition']].values
    
    bt_rmse, bt_per_axis = bilateral_trajectory_rmse(left_hand, right_hand)
    print(f"Bilateral Trajectory RMSE: {bt_rmse:.2f} units")
    print(f"  Per-axis: X={bt_per_axis[0]:.2f}, Y={bt_per_axis[1]:.2f}, Z={bt_per_axis[2]:.2f}")
    print(f"  → Lower = more symmetric movements")
    
    corr, corr_per_axis = correlation_joint_trajectories(left_hand, right_hand)
    print(f"\nCorrelation of Joint Trajectories: {corr:.3f}")
    print(f"  Per-axis: X={corr_per_axis[0]:.3f}, Y={corr_per_axis[1]:.3f}, Z={corr_per_axis[2]:.3f}")
    print(f"  → Values near +1 = highly coordinated bilateral control")
    
    # ========================================================================
    # SECTION 5.3: SPATIAL ENERGY METRICS
    # ========================================================================
    
    print("\n" + "="*80)
    print("5.3 SPATIAL ENERGY METRICS - Amplitude & Intensity")
    print("="*80)
    
    print("\n--- 5.3.1 Amplitude / Expansiveness ---")
    
    rom = joint_range_of_motion(right_hand_pos)
    print(f"Joint Range of Motion: {rom:.2f} spatial units")
    print(f"  → Larger = more expansive movement")
    
    vol = gesture_volume(right_hand_pos)
    print(f"\nGesture Volume: {vol:.2f} units³")
    print(f"  → Larger = greater workspace utilization")
    
    hull_vol = convex_hull_volume(right_hand_pos)
    print(f"\nConvex Hull Volume: {hull_vol:.2f} units³")
    print(f"  → Larger = more space-filling movement")
    
    print("\n--- 5.3.2 Energy / Intensity ---")
    
    qom = quantity_of_movement(right_hand_pos, frame_rate=frame_rate)
    print(f"Quantity of Movement: {qom:.2f} units/sec")
    print(f"  → Higher = more dynamic movement")
    
    # Weighted QoM across all joints
    wqom = weighted_qom(joint_positions, joint_weights=None)
    print(f"\nWeighted QoM (whole-body): {wqom:.2f} units/sec")
    print(f"  → Higher = more whole-body expressiveness")
    
    ke_mean = kinetic_energy_proxy(right_hand_pos, frame_rate, mass=1.0)
    print(f"\nKinetic Energy (proxy): Mean={ke_mean:.2f}")
    print(f"  → Higher = more vigorous movement")
    
    print("\n--- 5.3.3 Efficiency / Directness ---")
    
    directness = path_directness_index(right_hand_pos)
    print(f"Path Directness Index: {directness:.4f}")
    print(f"  → Closer to 1.0 = more efficient path")
    
    efficiency = energy_economy(right_hand_pos, frame_rate, task_performance=1.0)
    print(f"\nEnergy Economy: {efficiency:.4f}")
    print(f"  → Higher = more economical movement")
    
    # ========================================================================
    # SECTION 5.4: VARIABILITY & CONSISTENCY METRICS
    # ========================================================================
    
    print("\n" + "="*80)
    print("5.4 VARIABILITY & CONSISTENCY METRICS")
    print("="*80)
    
    # 5.4.1 Intra-subject, Intra-motion
    print("\n--- 5.4.1 Intra-subject, Intra-motion (within-trial consistency) ---")
    
    bpv = body_posture_variation(joint_positions)
    print(f"Body Posture Variation: {bpv:.2f} spatial units")
    print(f"  → Lower = more consistent postural control")
    
    print("\n--- Example: Multi-trial analysis (requires multiple recordings) ---")
    print("For demonstration, simulating 5 trials with added noise:")
    
    # Simulate 5 trials by adding noise to current trial
    n_trials = 5
    simulated_trials = []
    for i in range(n_trials):
        noise = np.random.normal(0, 5.0, right_hand_pos.shape)  # 5 unit std noise
        trial = right_hand_pos + noise
        simulated_trials.append(trial)
    
    # Convert to array for trajectory_variability
    simulated_trials = np.array(simulated_trials)
    
    # Trajectory Variability
    tv = trajectory_variability(simulated_trials)
    print(f"\nTrajectory Variability: {tv:.2f} spatial units")
    print(f"  → Lower = more reproducible movement paths")
    
    # Trial-to-trial SD (using SPARC as example metric)
    sparc_values = []
    for trial in simulated_trials:
        s = spectral_arc_length(trial, frame_rate=frame_rate)
        sparc_values.append(s)
    sparc_values = np.array(sparc_values)
    
    sd = trial_to_trial_sd(sparc_values)
    mean_sparc = np.mean(sparc_values)
    cv = (sd / abs(mean_sparc)) * 100 if mean_sparc != 0 else np.inf
    print(f"\nTrial-to-trial SD (SPARC metric): SD={sd:.4f}, CoV={cv:.2f}%")
    print(f"  → Lower = more consistent smoothness across trials")
    
    # 5.4.2 Inter-subject variability (example)
    print("\n--- 5.4.2 Inter-subject, Intra-motion (consistency across people) ---")
    print("Example: Simulating 10 subjects performing same movement:")
    
    # Simulate 10 subjects with different mean SPARC values
    subject_sparc_values = np.random.normal(-60, 10, 10)  # Mean=-60, SD=10
    cov = coefficient_of_variability(subject_sparc_values)
    mean_val = np.mean(subject_sparc_values)
    sd_val = np.std(subject_sparc_values, ddof=1)
    print(f"\nCoefficient of Variation (SPARC across subjects):")
    print(f"  Mean={mean_val:.2f}, SD={sd_val:.2f}, CoV={cov:.2f}%")
    print(f"  → Lower CoV = more consistent movement pattern across individuals")
    
    print("\n" + "-"*80)
    print("Note: The above examples use simulated multi-trial/multi-subject data.")
    print("For real analysis, load multiple BVH files and compute metrics for each.")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nFor detailed metric formulas and references, see individual module docstrings:")
    print("  - fluency.py: Smoothness and coordination metrics")
    print("  - balance_symmetry.py: Stability and bilateral coordination")
    print("  - spatial_energy.py: Amplitude and intensity metrics")
    print("  - variability.py: Reproducibility and consistency metrics")
    print("\n")


if __name__ == "__main__":
    main()
