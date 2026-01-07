"""
Balance & Symmetry Metrics Module

This module implements metrics for assessing postural stability, equilibrium control,
and bilateral movement coordination.

According to TABLE II from the paper (FG2026), these metrics include:
- Balance (Static): Sway Amplitude RMS, Mean Velocity of Sway, Sway Area
- Balance (Dynamic): Extrapolated CoM (XCoM), Margin of Stability (MoS), Local Dynamic Stability (Lyapunov)
- Symmetry (Spatiotemporal): Symmetry Index (SI)
- Symmetry (Kinematic/Kinetic): Bilateral Trajectory RMSE, Correlation of Joint Trajectories, GQoM

Data sources: Center of Mass trajectory, global joint positions
"""

import numpy as np
from scipy import stats


# ============================================================================
# BALANCE (STATIC) METRICS
# ============================================================================

def sway_amplitude_rms(com_trajectory, axis='both'):
    """
    Calculate Sway Amplitude RMS (SAmp).
    
    Shows how far the body's center shifts while standing still.
    Larger values mean more sway.
    
    Parameters
    ----------
    com_trajectory : ndarray, shape (n_frames, 3)
        Center of Mass trajectory with columns [x, y, z] in spatial units
    axis : str, default='both'
        - 'x': Mediolateral (ML) sway only
        - 'y': Vertical direction only
        - 'z': Anteroposterior (AP) sway only
        - 'both': Combined ML and AP sway (typical)
    
    Returns
    -------
    rms : float
        RMS displacement in spatial units
    
    Notes
    -----
    - Formula: SAmp = sqrt(1/N * Σ(r_i - r_mean)^2).
    
    References
    ----------
    Yamamoto, T., Smith, C. E., Suzuki, Y., Kiyono, K., Tanahashi, T., & Nomura, T.
    (2015). Universal and individual characteristics of postural sway during quiet
    standing in healthy young adults. PLoS ONE.
    """
    # Detrend: compute displacement from mean position
    com_mean = com_trajectory.mean(axis=0)
    displacement = com_trajectory - com_mean
    
    if axis == 'x':
        rms = np.sqrt(np.mean(displacement[:, 0]**2))
    elif axis == 'y':
        rms = np.sqrt(np.mean(displacement[:, 1]**2))
    elif axis == 'z':
        rms = np.sqrt(np.mean(displacement[:, 2]**2))
    elif axis == 'both':
        # Combined 2D planar sway
        rms_x = np.mean(displacement[:, 0]**2)
        rms_z = np.mean(displacement[:, 2]**2)
        rms = np.sqrt(rms_x + rms_z)
    else:
        raise ValueError("axis must be 'x', 'y', 'z', or 'both'")
    
    return rms


def mean_velocity_of_sway(com_trajectory, frame_rate=None, axis='both'):
    """
    Calculate Mean Velocity of Sway (MVS).
    
    Captures how fast the body sways over time.
    Higher values indicate less stability.
    
    Parameters
    ----------
    com_trajectory : ndarray, shape (n_frames, 3)
        Center of Mass trajectory [x, y, z] in spatial units
    frame_rate : float, optional
        Sampling frequency in Hz. If provided, returns velocity in units/second.
    axis : str, default='both'
        - 'x', 'y', 'z', or 'both' (planar).
    
    Returns
    -------
    mean_vel : float
        Mean velocity in spatial units per second (if frame_rate given) or per frame.
    
    Notes
    -----
    - Formula: MVS = 1/T * Σ ||p(t_i) - p(t_i-1)||.
    
    References
    ----------
    Lin, D., Nussbaum, M. A., Seol, H., & Madigan, M. L. (2008). Reliability of
    cop-based postural sway measures and age-related differences. Gait & Posture.
    """
    displacement = np.diff(com_trajectory, axis=0)
    
    if axis == 'x':
        dist = np.abs(displacement[:, 0])
    elif axis == 'y':
        dist = np.abs(displacement[:, 1])
    elif axis == 'z':
        dist = np.abs(displacement[:, 2])
    elif axis == 'both':
        dist = np.sqrt(displacement[:, 0]**2 + displacement[:, 2]**2)
    else:
        raise ValueError("axis must be 'x', 'y', 'z', or 'both'")
    
    mean_vel = dist.mean()
    
    if frame_rate is not None:
        mean_vel *= frame_rate
    
    return mean_vel


def sway_area_ellipse(com_trajectory, confidence=0.95):
    """
    Calculate Sway Area (SA) - 95% Confidence Ellipse.
    
    Represents the size of the area covered by body sway.
    Larger areas mean poorer balance.
    
    Parameters
    ----------
    com_trajectory : ndarray, shape (n_frames, 3)
        Center of Mass trajectory [x, y, z] in spatial units
    confidence : float, default=0.95
        Confidence level for ellipse (typically 0.95).
    
    Returns
    -------
    area : float
        Ellipse area in spatial units squared.
    
    Notes
    -----
    - Formula matches Table II: SA_p = π * chi^2 * sqrt(lambda1 * lambda2).
    
    References
    ----------
    Duarte, M., Freitas, S., & Zatsiorsky, V. (2011). Control of equilibrium in
    humans—sway over sway. Motor Control: Theories, Experiments, and Applications.
    
    Duarte, M., & Zatsiorsky, V. M. (2000). On the fractal properties of natural
    human standing. Neuroscience Letters.
    """
    # Extract horizontal plane coordinates (x, z) - assuming y is up
    com_2d = com_trajectory[:, [0, 2]]
    
    # Center the data
    com_centered = com_2d - com_2d.mean(axis=0)
    
    # Compute covariance matrix
    cov = np.cov(com_centered.T)
    
    # Eigenvalue decomposition
    eigenvalues, _ = np.linalg.eig(cov)
    
    # Chi-square value for 2D confidence ellipse (2 degrees of freedom)
    chi2_val = stats.chi2.ppf(confidence, df=2)
    
    # Semi-axes lengths (scaled by chi-square)
    a = np.sqrt(chi2_val * eigenvalues[0])
    b = np.sqrt(chi2_val * eigenvalues[1])
    
    # Ellipse area
    area = np.pi * a * b
    
    return area


# ============================================================================
# BALANCE (DYNAMIC) METRICS
# ============================================================================

def extrapolated_center_of_mass(com_position, com_velocity, body_height, gravity=9.81):
    """
    Calculate Extrapolated Center of Mass (XCoM).
    
    Accounts for both position and momentum of the body to predict if balance
    can be maintained.
    
    Parameters
    ----------
    com_position : ndarray, shape (n_frames, 3)
        CoM positions.
    com_velocity : ndarray, shape (n_frames, 3)
        CoM velocities.
    body_height : float
        Subject height (to estimate pendulum length l).
    
    Returns
    -------
    xcom : ndarray
        Extrapolated CoM trajectory.
    
    Notes
    -----
    - Formula: XCoM = x + v/ω₀
    - ω₀ = sqrt(g/l)
    
    References
    ----------
    Hof, A. L. (2007). The 'extrapolated center of mass' concept suggests a simple
    control of balance in walking. Gait & Posture.
    """
    # Estimate inverted pendulum natural frequency (omega_0)
    # l is effective pendulum length, often approximated by leg length or height factor
    pendulum_length = body_height 
    omega_0 = np.sqrt(gravity / pendulum_length)
    
    xcom = com_position + com_velocity / omega_0
    return xcom


def margin_of_stability(com_trajectory, com_velocity, foot_positions,
                        body_height=1.7, gravity=9.81):
    """
    Calculate Margin of Stability (MoS).
    
    Measures how far the body is from losing balance (distance between XCoM
    and Base of Support boundary). Bigger margins mean safer stability.
    
    Parameters
    ----------
    com_trajectory : ndarray, shape (n_frames, 3)
        CoM positions [x, y, z] in meters.
    com_velocity : ndarray, shape (n_frames, 3)
        CoM velocities [vx, vy, vz] in m/s.
    foot_positions : ndarray, shape (n_frames, 2, 3)
        Positions of feet [n_frames, [left, right], [x, y, z]].
    body_height : float
        Subject height (to estimate pendulum length l).
    gravity : float
        Gravity constant.
    
    Returns
    -------
    mos_ml, mos_ap : ndarray
        Margins of stability in ML and AP directions. Positive = Stable.
    
    Notes
    -----
    - Formula: MoS = u - XCoM (where u is BoS boundary)
    
    References
    ----------
    Hof, A. L., Gazendam, M. G. J., & Sinke, W. E. (2005). The condition for
    dynamic stability. Journal of Biomechanics.
    """
    # Calculate XCoM using the dedicated function
    xcom = extrapolated_center_of_mass(com_trajectory, com_velocity, body_height, gravity)
    
    # Base of support boundaries (simplified as foot positions)
    left_foot = foot_positions[:, 0, :]
    right_foot = foot_positions[:, 1, :]
    
    # Mediolateral: MoS = distance to lateral boundary
    # Stable if XCoM is between feet. MoS is min distance to either foot boundary.
    mos_ml = np.minimum(
        np.abs(xcom[:, 0] - left_foot[:, 0]),
        np.abs(xcom[:, 0] - right_foot[:, 0])
    )
    
    # Anteroposterior: MoS = distance to forward boundary (leading foot)
    # Assuming Z is AP direction
    leading_foot_ap = np.maximum(left_foot[:, 2], right_foot[:, 2])
    mos_ap = leading_foot_ap - xcom[:, 2]
    
    return mos_ml, mos_ap


def local_dynamic_stability(trajectory, frame_rate, embedding_dim=5,
                              time_delay=10, min_tsep=None):
    """
    Calculate Local Dynamic Stability (Lyapunov Exponent).
    
    Tells how sensitive gait is to small disturbances.
    Lower values mean more stable walking.
    
    Parameters
    ----------
    trajectory : ndarray, shape (n_frames, 3)
        CoM trajectory (or other state variable).
    frame_rate : float
        Sampling rate.
    embedding_dim : int, default=5
        Embedding dimension for phase space reconstruction
    time_delay : int, default=10
        Time delay (in frames) for embedding
    min_tsep : int, optional
        Minimum temporal separation for nearest neighbors
    
    Returns
    -------
    lambda_max : float
        Max Lyapunov exponent. Lower values = More stable.
    
    Notes
    -----
    - Formula: d(t) ~ e^(lambda * t)
    - Tracks divergence d_j(t) of nearest neighbors in state space.
    - Implementation based on Rosenstein's algorithm logic.
    
    References
    ----------
    Bruijn, S. M., Meijer, O. G., Beek, P. J., & van Dieën, J. H. (2013).
    Assessing the stability of human locomotion: a review of current measures.
    Journal of Biomechanics.
    """
    # Use ML-AP resultant for 1D time series proxy for stability
    # Or use dominant axis. Here we use planar magnitude.
    com_horiz = np.sqrt(trajectory[:, 0]**2 + trajectory[:, 2]**2)
    n_frames = len(com_horiz)
    
    # Time-delay embedding
    n_embedded = n_frames - (embedding_dim - 1) * time_delay
    if n_embedded < 10: return np.nan
    
    embedded = np.zeros((n_embedded, embedding_dim))
    for i in range(embedding_dim):
        start_idx = i * time_delay
        end_idx = start_idx + n_embedded
        embedded[:, i] = com_horiz[start_idx:end_idx]
    
    if min_tsep is None:
        min_tsep = int(0.1 * n_embedded)
    
    # Find nearest neighbors and track divergence
    n_points = embedded.shape[0]
    divergence = []
    
    for i in range(n_points - min_tsep):
        distances = np.linalg.norm(embedded - embedded[i], axis=1)
        valid_mask = np.ones(n_points, dtype=bool)
        # Exclude temporally close points
        valid_mask[max(0, i - min_tsep):min(n_points, i + min_tsep + 1)] = False
        
        if np.any(valid_mask):
            valid_distances = distances.copy()
            valid_distances[~valid_mask] = np.inf
            nn_idx = np.argmin(valid_distances)
            
            # Track divergence over time (e.g. 50 steps horizon)
            max_steps = min(n_points - i, n_points - nn_idx)
            for step in range(1, min(max_steps, 50)): 
                div = np.linalg.norm(embedded[i + step] - embedded[nn_idx + step])
                if div > 1e-6:
                    divergence.append((step, np.log(div)))
    
    # Linear fit to divergence curve
    if len(divergence) > 0:
        divergence = np.array(divergence)
        steps = divergence[:, 0]
        log_div = divergence[:, 1]
        # Fit slope (lambda)
        coeffs = np.polyfit(steps / frame_rate, log_div, 1)
        lambda_max = coeffs[0]
    else:
        lambda_max = np.nan
    
    return lambda_max


# ============================================================================
# SYMMETRY (SPATIOTEMPORAL) METRICS
# ============================================================================

def gait_symmetry_index(left_param, right_param):
    """
    Calculate Symmetry Index (SI).
    
    Compares left and right steps. Zero difference means perfect symmetry.
    
    Parameters
    ----------
    left_param : ndarray or float
        Parameter for left side (e.g., step length, time).
    right_param : ndarray or float
        Parameter for right side.
    
    Returns
    -------
    si : float
        Symmetry Index (percentage). 0% = Perfect Symmetry.
    
    Notes
    -----
    - Formula from Table II: SI = 100% * (X_L - X_R) / (0.5 * (X_L + X_R)).
    - WARNING: This is a normalized difference, not a ratio.
    
    References
    ----------
    Patterson, K. C., Gage, W. H., Brooks, D., Black, S. E., & McIlroy, W. E.
    (2010). Evaluation of gait symmetry after stroke: A comparison of current
    methods and recommendations for standardization. Gait & Posture.
    """
    val_left = np.mean(left_param)
    val_right = np.mean(right_param)
    
    numerator = val_left - val_right
    denominator = 0.5 * (val_left + val_right)
    
    if np.abs(denominator) < 1e-9:
        return np.nan
        
    si = 100.0 * (numerator / denominator)
    
    return si


# ============================================================================
# SYMMETRY (KINEMATIC/KINETIC) METRICS
# ============================================================================

def bilateral_trajectory_rmse(left_trajectory, right_trajectory):
    """
    Calculate Bilateral Trajectory RMSE (BT).
    
    Quantifies how far limb movements differ between sides.
    
    Parameters
    ----------
    left_trajectory : ndarray
        Left limb trajectory.
    right_trajectory : ndarray
        Right limb trajectory.
    
    Returns
    -------
    bt_rmse : float
        Overall bilateral trajectory RMSE. Lower values = Higher symmetry.
    bt_per_axis : ndarray, shape (3,)
        RMSE for each axis [X, Y, Z].
    
    Notes
    -----
    - Formula: BT = sqrt(1/N * Σ(x_L - x_R)^2).
    - Automatically mirrors the right trajectory along the X-axis (assuming 
      X is mediolateral) to allow comparison of symmetric movements.
    
    References
    ----------
    Cabral, S., Pinheiro, F., Rodrigues, J., Rodrigues, J., & Sousa, I. (2022).
    Reliability of a global gait symmetry index based on linear joint displacements.
    Applied Sciences.
    """
    if left_trajectory.shape != right_trajectory.shape:
        raise ValueError("Trajectories must have same shape")
    
    # Mirror right trajectory along X-axis for valid comparison in global frame
    # (Assuming index 0 is X/Mediolateral)
    right_mirrored = right_trajectory.copy()
    right_mirrored[:, 0] = -right_mirrored[:, 0]
    
    squared_diff = (left_trajectory - right_mirrored) ** 2
    bt_rmse = np.sqrt(np.mean(squared_diff))
    
    # Per-axis RMSE
    bt_per_axis = np.sqrt(np.mean(squared_diff, axis=0))
    
    return bt_rmse, bt_per_axis


def correlation_joint_trajectories(left_trajectory, right_trajectory):
    """
    Calculate Correlation of Joint Trajectories (CJT).
    
    Assesses how closely left and right limb movements follow the same pattern.
    
    Returns
    -------
    corr : float
        Overall correlation coefficient (-1 to 1). Higher = Greater symmetry.
    corr_per_axis : ndarray, shape (3,)
        Correlation coefficient for each axis [X, Y, Z].
    
    Notes
    -----
    - Values near +1: Highly correlated, symmetric movements
    - Values near 0: Uncorrelated, independent movements
    - Values near -1: Anti-correlated movements
    
    References
    ----------
    Sadeghi, H., Allard, P., Prince, F., & Labelle, H. (2000). Symmetry and limb
    dominance in able-bodied gait: a review. Gait & Posture.
    """
    # Overall correlation (flatten all dimensions)
    l_flat = left_trajectory.flatten()
    r_flat = right_trajectory.flatten()
    corr_matrix = np.corrcoef(l_flat, r_flat)
    corr = corr_matrix[0, 1]
    
    # Per-axis correlation
    corr_per_axis = np.zeros(3)
    for i in range(3):
        corr_matrix_axis = np.corrcoef(left_trajectory[:, i], right_trajectory[:, i])
        corr_per_axis[i] = corr_matrix_axis[0, 1]
    
    return corr, corr_per_axis


def generalized_quantity_of_movement(joint_positions, joint_weights=None, normalize=True):
    """
    Calculate Generalized Quantity of Movement (GQoM).
    
    Indicates overall body motion; can reveal imbalances if analyzed per side.
    
    Parameters
    ----------
    joint_positions : ndarray, shape (n_frames, n_joints, 3)
        3D positions of joints.
    joint_weights : ndarray, optional
        Weights for each joint (e.g. based on mass). Default is equal weights.
    normalize : bool
        If True, divide by number of joints (Average QoM).
    
    Returns
    -------
    gqom : float
        Total movement quantity.
    
    Notes
    -----
    - Formula: GQoM = (1/lambda) * Σ ||p_ij - p_(i-1)j||.
    
    References
    ----------
    Vatavu, R.-D. (2017). Beyond features for recognition: human-readable measures
    to understand users' whole-body gesture performance.
    International Journal of Human-Computer Interaction.
    """
    n_frames, n_joints, n_dims = joint_positions.shape
    
    if joint_weights is None:
        joint_weights = np.ones(n_joints)
        
    # Frame-to-frame displacement
    displacements = np.diff(joint_positions, axis=0)
    magnitudes = np.linalg.norm(displacements, axis=2)
    
    # Weight each joint's contribution
    weighted_displacements = magnitudes * joint_weights[np.newaxis, :]
    
    # Sum across all joints and frames
    total_qom = np.sum(weighted_displacements)
    
    if normalize:
        return total_qom / n_joints
    
    return total_qom