"""
Data Loading and Preprocessing Module

This module handles BVH file loading, local rotation extraction, global position computation,
and Center of Mass calculation using Winter's anthropometric model.

Functions:
- load_bvh(): Parse BVH file and extract motion data
- extract_rotations(): Extract local rotations (Euler angles)
- compute_global_positions(): Forward kinematics to compute joint positions
- calculate_com(): Compute true Center of Mass using Winter's model
"""

import numpy as np
import pandas as pd
import warnings
from pymo.parsers import BVHParser
from pymo.preprocessing import MocapParameterizer

# Suppress PyMO performance warnings
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)


def load_bvh(filepath):
    """
    Parse BVH file and extract motion capture data.
    
    Parameters
    ----------
    filepath : str
        Path to BVH file
    
    Returns
    -------
    parsed_data : pymo.MocapData
        Parsed BVH data with skeleton hierarchy and motion channels
    mocap_data : pd.DataFrame
        DataFrame with all channels (positions for root, rotations for joints)
    frame_rate : float
        Sampling frequency in Hz
    unique_joints : list
        List of joint names in the skeleton
    
    Notes
    -----
    BVH files contain:
    - Skeleton hierarchy (joint names, offsets, parent-child relationships)
    - Motion data (rotation channels per joint, typically in Euler angles)
    - Frame time and total number of frames
    """
    parser = BVHParser()
    parsed_data = parser.parse(filepath)
    
    # Extract DataFrame representation
    mocap_data = parsed_data.values.reset_index().iloc[:, 1:]
    
    # Get frame rate
    frame_time = parsed_data.framerate
    frame_rate = 1.0 / frame_time
    
    # Extract joint names
    joint_names = [col.replace('_Xrotation', '').replace('_Yrotation', '').replace('_Zrotation', '')
                   .replace('_Xposition', '').replace('_Yposition', '').replace('_Zposition', '')
                   for col in mocap_data.columns]
    unique_joints = sorted(list(set(joint_names)))
    
    n_frames = len(mocap_data)
    n_channels = len(mocap_data.columns)
    
    print(f"✓ Loaded: {filepath}")
    print(f"  Frames: {n_frames}")
    print(f"  Channels: {n_channels}")
    print(f"  Joints: {len(unique_joints)}")
    print(f"  Frame rate: {frame_rate:.2f} Hz")
    
    return parsed_data, mocap_data, frame_rate, unique_joints


def extract_rotations(mocap_data):
    """
    Extract local rotations (Euler angles) from motion data.
    
    Local rotations represent joint angles in the skeleton's local coordinate system.
    These are the raw rotation values stored in the BVH file.
    
    Parameters
    ----------
    mocap_data : pd.DataFrame
        Motion capture data with all channels
    
    Returns
    -------
    local_rotations : pd.DataFrame
        DataFrame with only rotation channels (Xrotation, Yrotation, Zrotation)
    
    Notes
    -----
    - Used for fluency metrics (smoothness analysis via angular velocity/acceleration)
    - Indicate joint coordination patterns
    - Domain-agnostic: work across different skeleton scales
    - Representation: Euler angles in degrees (XYZ order)
    """
    rotation_columns = [col for col in mocap_data.columns if 'rotation' in col.lower()]
    local_rotations = mocap_data[rotation_columns]
    
    n_rotation_channels = len(rotation_columns)
    n_joints_with_rotation = n_rotation_channels // 3
    
    print(f"\n✓ Extracted local rotations")
    print(f"  Rotation channels: {n_rotation_channels}")
    print(f"  Joints with rotation: {n_joints_with_rotation}")
    
    return local_rotations


def compute_global_positions(parsed_data):
    """
    Compute global 3D positions for all joints via forward kinematics.
    
    Global positions represent joint locations in world coordinate space.
    Unlike local rotations, these account for the full kinematic chain from root to end-effector.
    
    Parameters
    ----------
    parsed_data : pymo.MocapData
        Parsed BVH data with skeleton hierarchy
    
    Returns
    -------
    global_positions : pd.DataFrame
        DataFrame with global (X, Y, Z) positions for every joint at every frame
    
    Notes
    -----
    Method: Forward kinematics via MocapParameterizer('position')
    - Applies skeleton hierarchy and joint offsets
    - Computes cumulative transformations through the kinematic chain
    - Produces 3D world coordinates for every joint
    
    Why global positions are needed:
    - Essential for balance metrics (e.g., Center of Mass, sway analysis)
    - Required for symmetry metrics (e.g., bilateral trajectory comparison)
    - Enable spatial energy metrics (e.g., amplitude, workspace volume)
    
    Reference:
    Winter, D. A. (2009). Biomechanics and Motor Control of Human Movement (4th ed.). Wiley.
    """
    mp = MocapParameterizer('position')
    positions_data = mp.fit_transform([parsed_data])
    global_positions = positions_data[0].values
    
    n_position_channels = len(global_positions.columns)
    n_joints_with_position = n_position_channels // 3
    
    print(f"\n✓ Computed global positions via forward kinematics")
    print(f"  Position channels: {n_position_channels}")
    print(f"  Joints with positions: {n_joints_with_position}")
    
    return global_positions


def calculate_com(positions_df):
    """
    Compute true Center of Mass using Winter's anthropometric model.
    
    This function calculates whole-body center of mass by taking a weighted sum of body
    segment positions using empirically determined mass fractions.
    
    Parameters
    ----------
    positions_df : pd.DataFrame
        DataFrame with global positions (X, Y, Z) for all joints
    
    Returns
    -------
    com_trajectory : np.ndarray
        Array of shape (n_frames, 3) containing CoM position at each frame
    
    Notes
    -----
    Important distinction:
    - Hips position ≠ True Center of Mass
    - The Hips joint is only an approximation of body's CoM
    - True CoM requires weighted contribution from all body segments
    
    Winter's Segment Mass Distribution:
    - Head: 8.1%, Trunk: 50.7%
    - Upper arms: 2.8% each, Forearms: 1.6% each, Hands: 0.6% each
    - Thighs: 10.0% each, Shanks: 4.65% each, Feet: 1.45% each
    
    ⚠️ IMPORTANT: Adjust joint names in segment_masses dict for your BVH file.
    Different BVH files use different naming conventions.
    
    References
    ----------
    Winter, D. A. (2009). Biomechanics and Motor Control of Human Movement (4th ed.).
    Wiley. https://doi.org/10.1002/9780470549148
    """
    # Winter's segment mass percentages (relative to total body mass)
    segment_masses = {
        'Head': 0.081,
        'Spine': 0.507,
        'RightArm': 0.028,
        'LeftArm': 0.028,
        'RightForeArm': 0.016,
        'LeftForeArm': 0.016,
        'RightHand': 0.006,
        'LeftHand': 0.006,
        'RightUpLeg': 0.100,
        'LeftUpLeg': 0.100,
        'RightLeg': 0.0465,
        'LeftLeg': 0.0465,
        'RightFoot': 0.0145,
        'LeftFoot': 0.0145
    }
    
    n_frames = len(positions_df)
    com_trajectory = np.zeros((n_frames, 3))
    
    total_mass_accounted = 0.0
    missing_segments = []
    
    # Weighted sum of segment positions
    for segment_name, mass_fraction in segment_masses.items():
        x_col = f'{segment_name}_Xposition'
        y_col = f'{segment_name}_Yposition'
        z_col = f'{segment_name}_Zposition'
        
        if x_col in positions_df.columns:
            segment_pos = positions_df[[x_col, y_col, z_col]].values
            com_trajectory += mass_fraction * segment_pos
            total_mass_accounted += mass_fraction
        else:
            missing_segments.append(f"{segment_name} ({mass_fraction*100:.1f}%)")
    
    # Warning if not all segments found
    if missing_segments:
        print(f"\n  ⚠️  WARNING: Missing segments in BVH skeleton:")
        print(f"      {', '.join(missing_segments)}")
        print(f"      Total mass accounted: {total_mass_accounted*100:.1f}%")
        print(f"      → CoM will be approximate. Update joint names in segment_masses dict.")
        
        if total_mass_accounted > 0:
            com_trajectory = com_trajectory / total_mass_accounted
    
    print(f"\n✓ Computed Center of Mass using Winter's anthropometric model")
    print(f"  CoM trajectory shape: {com_trajectory.shape}")
    print(f"  Mean vertical position (Y): {np.mean(com_trajectory[:, 1]):.2f} spatial units")
    
    return com_trajectory
