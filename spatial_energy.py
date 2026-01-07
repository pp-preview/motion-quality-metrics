"""
Spatial Energy Metrics Module

This module implements metrics for assessing movement amplitude, workspace utilization,
and dynamic intensity.

According to TABLE III from the paper (FG2026), these metrics include:
- Amplitude / Expansiveness:
    - Gesture Volume (GV)
    - Joint Range of Motion (ROM)
    - Reach / Extensiveness Index (RI)
    - Convex Hull Volume (CHV)
- Energy / Intensity:
    - Quantity of Movement (QoM)
    - Weighted Quantity of Movement (WQoM)
    - Kinetic-Energy Proxy (KEP)
- Efficiency / Directness:
    - Path Directness Index (PDI)
    - Trajectory Curvature Index (TCI)
    - Energy Economy (EE)

Data source: Global joint positions
"""

import numpy as np
from scipy.spatial import ConvexHull


# ============================================================================
# AMPLITUDE / EXPANSIVENESS METRICS
# ============================================================================

def gesture_volume(trajectory):
    """
    Calculate Gesture Volume (GV) - Bounding Box Volume.
    
    The bounding box of movement space; larger values mean more expansive 
    movement but may include empty space.
    
    Parameters
    ----------
    trajectory : ndarray, shape (n_frames, 3)
        Position trajectory [x, y, z].
    
    Returns
    -------
    volume : float
        Bounding box volume.
    
    References
    ----------
    Vatavu, R.-D. (2017). Beyond features for recognition: human-readable measures
    to understand users' whole-body gesture performance. International Journal of
    Human-Computer Interaction.
    """
    mins = np.min(trajectory, axis=0)
    maxs = np.max(trajectory, axis=0)
    ranges = maxs - mins
    
    # Volume = product of ranges in x, y, z
    volume = np.prod(ranges)
    
    return volume


def joint_range_of_motion(trajectory):
    """
    Calculate Joint Range of Motion (ROM).
    
    How much a joint angle (or position) changes; higher values mean more 
    flexible or expansive motion.
    
    Parameters
    ----------
    trajectory : ndarray, shape (n_frames, 3) or (n_frames, n_joints, 3)
        Position trajectory or angle trajectory.
        If 3D array (joints), computes mean ROM across joints.
    
    Returns
    -------
    rom : float
        Range of motion.
    
    References
    ----------
    Collins, S. H., Adamczyk, P. G., & Kuo, A. D. (2009). Dynamic arm swinging
    in human walking. Proceedings of the Royal Society B: Biological Sciences.
    """
    # If input is multiple joints (n_frames, n_joints, n_dims)
    if trajectory.ndim == 3:
        # Calculate range for each joint/dimension and average
        # Ideally ROM is max - min per axis/angle
        ranges = np.max(trajectory, axis=0) - np.min(trajectory, axis=0)
        # Average across joints and dimensions (simplified scalar)
        rom = np.mean(ranges)
    else:
        # Single trajectory
        ranges = np.max(trajectory, axis=0) - np.min(trajectory, axis=0)
        rom = np.linalg.norm(ranges) # Diagonal or mean
        
    return rom


def reach_extensiveness_index(limb_trajectory, trunk_trajectory):
    """
    Calculate Reach / Extensiveness Index (RI).
    
    How far a limb extends from the trunk. Higher values mean more open,
    outward movement.
    
    Parameters
    ----------
    limb_trajectory : ndarray, shape (n_frames, 3)
        Position of the limb effector (e.g., wrist).
    trunk_trajectory : ndarray, shape (n_frames, 3)
        Position of the trunk/root (e.g., sternum or spine base).
    
    Returns
    -------
    ri : float
        Average euclidean distance between limb and trunk.
    
    Notes
    -----
    Formula: RI = (1/N) * Σ ||x_limb(t) - x_trunk(t)||
    
    References
    ----------
    Camurri, A., Lagerlöf, I., & Volpe, G. (2003). Toward real-time multimodal
    recognition of expressive human gestures.
    
    Camurri, A., Mazzarino, B., & Volpe, G. (2008). Modelling and Analysing
    Expressive Gesture in Multimodal Systems.
    """
    # Calculate euclidean distance at each frame
    distances = np.linalg.norm(limb_trajectory - trunk_trajectory, axis=1)
    
    # Average distance over time
    ri = np.mean(distances)
    
    return ri


def convex_hull_volume(trajectory):
    """
    Calculate Convex Hull Volume (CHV).
    
    The true occupied space of the movement. Larger values mean denser, 
    more spread gestures.
    
    Parameters
    ----------
    trajectory : ndarray, shape (n_frames, 3) or (n_frames, n_joints, 3)
        Position trajectory.
    
    Returns
    -------
    volume : float
        Convex hull volume. Returns NaN if not enough points (<4) or coplanar.
    
    References
    ----------
    Larboulette, C., & Gibet, S. (2015). A review of computable expressive
    descriptors of human motion.
    """
    # Handle multiple joints by flattening to a single point cloud
    if trajectory.ndim == 3:
        points = trajectory.reshape(-1, 3)
    else:
        points = trajectory
    
    # Need at least 4 points for 3D hull
    if len(points) < 4:
        return np.nan
    
    try:
        hull = ConvexHull(points)
        volume = hull.volume
    except Exception:
        # Fails if points are coplanar
        volume = np.nan
    
    return volume


# ============================================================================
# ENERGY / INTENSITY METRICS
# ============================================================================

def quantity_of_movement(trajectory, frame_rate=None):
    """
    Calculate Quantity of Movement (QoM).
    
    Totals body movement speed; higher values mean more dynamic performance.
    
    Parameters
    ----------
    trajectory : ndarray, shape (n_frames, 3)
        Position trajectory.
    frame_rate : float, optional
        Used to normalize time (dt).
    
    Returns
    -------
    qom : float
        Sum of velocity magnitudes (integral of speed).
    
    References
    ----------
    Vatavu, R.-D. (2017). Beyond features for recognition: human-readable measures
    to understand users' whole-body gesture performance.
    International Journal of Human-Computer Interaction.
    """
    # Velocity = distance / dt (or just distance sum if frame_rate not needed for integral)
    # The formula in table is Integral ||v(t)|| dt.
    # Discretely: Sum(distance)
    displacements = np.diff(trajectory, axis=0)
    distances = np.linalg.norm(displacements, axis=1)
    qom = np.sum(distances)
    
    return qom


def weighted_qom(joint_positions, joint_weights=None):
    """
    Calculate Weighted Quantity of Movement (WQoM).
    
    Same as QoM but mass-weighted; higher values mean greater physical effort.
    
    Parameters
    ----------
    joint_positions : ndarray, shape (n_frames, n_joints, 3)
        3D positions of all joints.
    joint_weights : ndarray, optional
        Mass/Importance weights for each joint (m_j). Default is 1.0.
    
    Returns
    -------
    wqom : float
        Weighted QoM.
    
    References
    ----------
    Vatavu, R.-D. (2017). Beyond features for recognition.
    
    Chadefaux, D., Wanderley, M. M., & Palmer, C. (2012). Gestural control of
    musical expression in piano performance.
    """
    n_frames, n_joints, n_dims = joint_positions.shape
    
    if joint_weights is None:
        joint_weights = np.ones(n_joints)
    
    # Calculate QoM for each joint
    total_weighted_qom = 0.0
    for j in range(n_joints):
        traj = joint_positions[:, j, :]
        displacements = np.diff(traj, axis=0)
        dist = np.sum(np.linalg.norm(displacements, axis=1))
        
        # WQoM = Sum( mass_j * QoM_j )
        total_weighted_qom += joint_weights[j] * dist
    
    return total_weighted_qom


def kinetic_energy_proxy(trajectory, frame_rate, mass=1.0):
    """
    Calculate Kinetic-Energy Proxy (KEP).
    
    Approximates movement energy; higher values mean more forceful, vigorous motion.
    
    Parameters
    ----------
    trajectory : ndarray, shape (n_frames, 3)
        Position trajectory.
    frame_rate : float
        Sampling frequency.
    mass : float
        Mass constant.
    
    Returns
    -------
    kep_mean : float
        Average kinetic energy over time.
    
    References
    ----------
    Wong, J. D., Cluff, T., & Kuo, A. D. (2021). The energetic basis for smooth
    human arm movements. Proceedings of the National Academy of Sciences.
    """
    # Velocity
    velocity = np.diff(trajectory, axis=0) * frame_rate
    speed_sq = np.sum(velocity**2, axis=1)
    
    # KEP(t) = 0.5 * m * v^2
    kep_t = 0.5 * mass * speed_sq
    
    # Return mean KEP
    return np.mean(kep_t)


# ============================================================================
# EFFICIENCY / DIRECTNESS METRICS
# ============================================================================

def path_directness_index(trajectory):
    """
    Calculate Path Directness Index (PDI).
    
    Compares path to a straight line; values near 1 mean direct and efficient motion.
    
    Returns
    -------
    pdi : float
        Ratio L_actual / L_straight (Note: Table formula is L_actual / L_straight,
        which gives values >= 1. Usually PDI is L_straight/L_actual for 0-1.
        Following Table III formula strictly).
    
    References
    ----------
    Cavallo, A., et al. (2014). A biomechanical analysis of surgeon's gesture.
    IEEE Haptics Symposium.
    """
    straight_line = np.linalg.norm(trajectory[-1] - trajectory[0])
    
    displacements = np.diff(trajectory, axis=0)
    path_length = np.sum(np.linalg.norm(displacements, axis=1))
    
    if straight_line <= 0:
        return np.inf
        
    # Table III formula: PDI = L_actual / L_straight
    # (Note: This means 1 = direct, >1 = indirect)
    pdi = path_length / straight_line
    
    return pdi


def trajectory_curvature_index(trajectory, frame_rate):
    """
    Calculate Trajectory Curvature Index (TCI).
    
    Averages path bending. Lower values mean straighter, more economical trajectories.
    
    Parameters
    ----------
    trajectory : ndarray, shape (n_frames, 3)
        Position trajectory.
    frame_rate : float
        Sampling frequency.
    
    Returns
    -------
    tci : float
        Average curvature per unit length.
    
    Notes
    -----
    Formula: TCI = (1/L) * Integral(kappa(s) ds)
    kappa(s) = ||v x a|| / ||v||^3
    
    References
    ----------
    Brown, G. L., Seethapathi, N., & Srinivasan, M. (2020). A unified energy-
    optimality criterion predicts human navigation paths and speeds.
    Proceedings of the National Academy of Sciences.
    """
    dt = 1.0 / frame_rate
    
    # Velocity and Acceleration
    vel = np.diff(trajectory, axis=0) / dt
    acc = np.diff(vel, axis=0) / dt
    
    # Match lengths (drop last velocity point to match acceleration)
    vel = vel[:-1]
    
    # Compute curvature kappa at each point
    # kappa = ||v x a|| / ||v||^3
    cross_prod = np.cross(vel, acc)
    cross_norm = np.linalg.norm(cross_prod, axis=1)
    vel_norm = np.linalg.norm(vel, axis=1)
    
    # Avoid division by zero
    valid_idx = vel_norm > 1e-6
    kappa = np.zeros_like(cross_norm)
    kappa[valid_idx] = cross_norm[valid_idx] / (vel_norm[valid_idx]**3)
    
    # Path length L
    displacements = np.diff(trajectory, axis=0)
    L = np.sum(np.linalg.norm(displacements, axis=1))
    
    # Integrate kappa ds -> Sum(kappa * ds)
    # ds approx ||v|| * dt
    ds = vel_norm * dt
    integral_kappa = np.sum(kappa * ds)
    
    if L > 0:
        tci = integral_kappa / L
    else:
        tci = 0.0
        
    return tci


def energy_economy(trajectory, frame_rate, task_performance=1.0, mass=1.0):
    """
    Calculate Energy Economy (EE).
    
    Relates performance to energy cost; higher values mean more efficient movement.
    
    Parameters
    ----------
    trajectory : ndarray
        Position trajectory.
    frame_rate : float
        Sampling frequency.
    task_performance : float
        Task-specific measure (e.g. speed x accuracy). Default 1.0 if unknown.
    mass : float
        Mass for kinetic energy proxy.
    
    Returns
    -------
    ee : float
        Ratio of Task Performance / Total Energy.
    
    References
    ----------
    Wong, J. D., Cluff, T., & Kuo, A. D. (2021). The energetic basis for smooth
    human arm movements. Proceedings of the National Academy of Sciences.
    """
    # Total Energy E_tot = Integral E(t) dt
    kep_mean = kinetic_energy_proxy(trajectory, frame_rate, mass)
    duration = (len(trajectory) - 1) / frame_rate
    total_energy = kep_mean * duration
    
    if total_energy < 1e-9:
        return np.nan
        
    # EE = Task Performance / Total Energy
    ee = task_performance / total_energy
    
    return ee