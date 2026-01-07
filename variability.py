"""
Variability & Consistency Metrics Module

This module implements metrics for assessing reproducibility and stability across
repetitions, tasks, or individuals.

According to TABLE IV from the paper (FG2026), these metrics are categorized as:
- Intra-subject, Intra-motion:
    - Body Posture Variation (BPV)
    - Trajectory Variability (TV)
    - Trial-to-trial SD
- Intra-subject, Inter-motion:
    - Consistency SD
- Inter-subject, Intra-motion:
    - Coefficient of Variability (CoV)

Data source: Requires multiple trials or subjects; operates on metrics from other modules.
"""

import numpy as np

# ============================================================================
# INTRA-SUBJECT, INTRA-MOTION REPRODUCIBILITY
# ============================================================================

def body_posture_variation(positions):
    """
    Calculate Body Posture Variation (BPV).
    
    Average deviation of body postures from the centroid posture of the whole
    body gesture. Higher values mean larger dispersion.
    
    Parameters
    ----------
    positions : ndarray, shape (n_frames, n_joints, 3)
        Array of joint positions over time for a single trial.
    
    Returns
    -------
    bpv : float
        Body Posture Variation.
    
    Notes
    -----
    Formula: BPV = (1/n) * Sum_i Sum_j ||p_ij - p_mean_j||
    
    References
    ----------
    Vatavu, R.-D. (2017). Beyond features for recognition: human-readable measures
    to understand users' whole-body gesture performance.
    International Journal of Human-Computer Interaction.
    """
    n_frames, n_joints, n_dims = positions.shape
    
    # Centroid posture (mean across frames)
    mean_posture = np.mean(positions, axis=0)
    
    # Deviation of each frame from centroid
    deviations = positions - mean_posture
    distances = np.linalg.norm(deviations, axis=2)
    
    # Sum over joints and frames, averaged by number of frames (n)
    # Note: Table formula sums over i=1..n and j=1..J, then multiplies by 1/n.
    bpv = np.sum(distances) / n_frames
    
    return bpv


def trajectory_variability(trajectories):
    """
    Calculate Trajectory Variability (TV).
    
    Dispersion of joint/end-effector paths across repetitions.
    Higher TV means unstable execution.
    
    Parameters
    ----------
    trajectories : ndarray, shape (n_trials, n_frames, 3)
        Multiple trajectory trials, time-normalized to same length.
    
    Returns
    -------
    tv : float
        Trajectory Variability.
    
    Notes
    -----
    Formula: TV = (1/T) * Sum_t sqrt( (1/(N-1)) * Sum_i (x_i(t) - x_mean(t))^2 )
    Essentially the average pointwise standard deviation across time.
    
    References
    ----------
    Müller, H., & Sternad, D. (2004). Decomposition of variability in the
    execution of goal-oriented tasks. Journal of Experimental Psychology.
    """
    if trajectories.ndim != 3: return np.nan
    n_trials, n_frames, n_dims = trajectories.shape
    
    if n_trials < 2: return np.nan
    
    # Calculate SD at each time step (pointwise variability)
    # ddof=1 for sample standard deviation (N-1)
    pointwise_sd = np.std(trajectories, axis=0, ddof=1) # shape (n_frames, 3)
    
    # Combine spatial dimensions (usually Euclidean norm of SD vector or mean SD)
    # The formula implies scalar x_i(t), for 3D we typically take norm of dispersion
    spatial_dispersion = np.linalg.norm(pointwise_sd, axis=1)
    
    # Average over time T
    tv = np.mean(spatial_dispersion)
    
    return tv


def trial_to_trial_sd(metric_values):
    """
    Calculate Trial-to-trial SD.
    
    Reproducibility of scalar kinematic parameters across repetitions.
    Lower values mean more consistent repetition.
    
    Parameters
    ----------
    metric_values : ndarray
        Array of scalar values from N trials (e.g. peak velocities).
    
    Returns
    -------
    sd : float
        Standard deviation.
    
    References
    ----------
    Schwarz, A., et al. (2019). Systematic review on kinematic assessments of
    upper limb movements after stroke. Stroke.
    """
    # Formula: sqrt( 1/(N-1) * Sum(z_i - z_mean)^2 )
    return np.std(metric_values, ddof=1)


# ============================================================================
# INTRA-SUBJECT, INTER-MOTION REPRODUCIBILITY
# ============================================================================

def consistency_sd(errors_per_gesture):
    """
    Calculate Consistency SD.
    
    Reproducibility of task outcomes across gestures/conditions.
    Low = stable performance, High = variable.
    
    Parameters
    ----------
    errors_per_gesture : ndarray
        Outcome errors (or metric values) for M different gestures/conditions.
    
    Returns
    -------
    consistency : float
        Standard deviation across conditions.
    
    References
    ----------
    van Beers, R. J. (2013). How does our motor system determine its learning
    rate? PLOS ONE.
    """
    # Formula: sqrt( 1/(M-1) * Sum(e_i - e_mean)^2 )
    return np.std(errors_per_gesture, ddof=1)


# ============================================================================
# INTER-SUBJECT, INTRA-MOTION REPRODUCIBILITY
# ============================================================================

def coefficient_of_variability(metric_values_across_subjects):
    """
    Calculate Coefficient of Variability (CoV).
    
    Variability of a parameter (e.g. stride time) across subjects.
    Lower CoV = more consistent group performance.
    
    Parameters
    ----------
    metric_values_across_subjects : ndarray
        Metric values for multiple subjects.
    
    Returns
    -------
    cov : float
        Coefficient of variation (percentage).
    
    Notes
    -----
    Formula: CoV = 100 * (sigma_y / mu_y)
    
    References
    ----------
    Owings, T. M., & Grabiner, M. D. (2000). Step variability is increased
    in the elderly and in persons with parkinson's disease.
    Journal of Gerontology.
    """
    sigma = np.std(metric_values_across_subjects, ddof=1)
    mu = np.mean(metric_values_across_subjects)
    
    if mu == 0: return np.inf
    return 100.0 * (sigma / mu)