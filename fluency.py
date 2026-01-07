"""
Fluency Metrics Module

This module implements metrics for assessing movement fluency - smoothness, coordination,
and continuous flow of movement.

According to TABLE I from the paper, Fluency includes:
- Temporal Smoothness: Log Dimensionless Jerk, Spectral Arc Length, Local Maxima
- Postural Continuity: Body Posture Rate
- Spatial Regularity: Curvature Variability, Tortuosity
- Rhythmic Consistency: Harmonic Ratio, Velocity Autocorrelation Decay

Data sources: Local rotations (joint angles), global positions (trajectories)
"""

import numpy as np
from scipy import signal


# ============================================================================
# TEMPORAL SMOOTHNESS METRICS
# ============================================================================

def log_dimensionless_jerk(trajectory, frame_rate):
    """
    Calculate log-transformed dimensionless jerk for movement smoothness assessment.

    This metric quantifies movement fluency by penalizing abrupt changes in the
    *speed profile* v(t)=||dx/dt|| through the squared second derivative of speed.
    A logarithmic transform is applied to improve numerical stability and
    distributional properties.

    IMPORTANT INTERPRETATION:
    - The paper defines DLJ as negative; here we return the log magnitude.
    - Higher values (less negative in paper, or higher magnitude here depending on sign convention)
      generally indicate smoother movement.
    - In this implementation: Higher output = Smoother movement.

    Parameters
    ----------
    trajectory : ndarray, shape (n_frames, D)
        Position trajectory in spatial units.
    frame_rate : float
        Sampling frequency in Hz.

    Returns
    -------
    ldlj : float
        Log dimensionless jerk. Higher values = smoother movement.

    Notes
    -----
    - Computed on the scalar speed signal v(t), not directly on position.
    - Uses the second derivative of speed with respect to time.
    - Normalized by movement duration and peak speed to be scale-independent.
    - Formula corresponds to -ln|DLJ| from Table I.

    References
    ----------
    Balasubramanian, S., Melendez-Calderon, A., Roby-Brami, A., & Burdet, E. (2015).
    On the analysis of movement smoothness. Journal of NeuroEngineering and Rehabilitation.
    """
    dt = 1.0 / frame_rate

    if trajectory is None or len(trajectory) < 4:
        return np.nan

    # Velocity (vector) and speed (scalar)
    velocity = np.diff(trajectory, axis=0) / dt
    speed = np.linalg.norm(velocity, axis=1)

    if len(speed) < 3:
        return np.nan

    # Second derivative of speed
    d2v = np.diff(speed, n=2) / (dt ** 2)

    duration = (len(speed) - 1) * dt
    v_peak = np.max(speed)

    if duration <= 0 or v_peak <= 0:
        return np.nan

    # Integral of squared second derivative
    integral = np.sum(d2v ** 2) * dt

    dj = (duration ** 5 / (v_peak ** 2)) * integral
    if dj <= 0 or not np.isfinite(dj):
        return np.nan

    return -np.log(dj)


def spectral_arc_length(trajectory, frame_rate, padlevel=4, fc=10.0):
    """
    Calculate Spectral Arc Length (SPARC) for movement smoothness.
    
    SPARC quantifies smoothness by measuring the arc length of the normalized magnitude
    spectrum of the speed profile. Smoother movements have more concentrated spectral
    energy at lower frequencies, resulting in more negative SPARC values.
    
    Parameters
    ----------
    trajectory : ndarray, shape (n_frames, 3)
        Position trajectory [x, y, z] in spatial units
    frame_rate : float
        Sampling frequency in Hz
    padlevel : int, default=4
        Zero-padding level for FFT (2^padlevel points)
    fc : float, default=10.0
        Maximum frequency for arc length integration in Hz
        (Note: 10Hz corresponds to approx 20pi rad/s as recommended in Table I).
    
    Returns
    -------
    sparc : float
        Spectral arc length. More negative values = smoother movement.
    
    Notes
    -----
    - Formula: SPARC = -∫ sqrt(1 + (dM/dω)²) dω, where M is normalized magnitude spectrum
    - Uses FFT to compute frequency spectrum of speed profile
    - Integrates up to cutoff frequency fc
    - More negative values indicate smoother, more ballistic movements
    
    References
    ----------
    Balasubramanian, S., Melendez-Calderon, A., Roby-Brami, A., & Burdet, E. (2015).
    On the analysis of movement smoothness. Journal of NeuroEngineering and Rehabilitation.
    """
    # Compute speed profile
    velocity = np.diff(trajectory, axis=0) * frame_rate
    speed = np.linalg.norm(velocity, axis=1)
    
    # Zero-pad for better frequency resolution
    nfft = 2 ** padlevel * len(speed)
    
    # Compute FFT
    spectrum = np.fft.fft(speed, n=nfft)
    freq = np.fft.fftfreq(nfft, d=1.0/frame_rate)
    
    # Take positive frequencies only
    pos_freq_idx = freq >= 0
    freq = freq[pos_freq_idx]
    magnitude = np.abs(spectrum[pos_freq_idx])
    
    # Normalize magnitude spectrum
    magnitude_normalized = magnitude / magnitude[0] if magnitude[0] > 0 else magnitude
    
    # Find index corresponding to cutoff frequency
    fc_idx = np.argmin(np.abs(freq - fc))
    
    # Compute arc length up to fc
    mag_norm = magnitude_normalized[:fc_idx]
    omega = 2 * np.pi * freq[:fc_idx]
    
    # Derivative of magnitude w.r.t. frequency
    dmag_domega = np.diff(mag_norm) / np.diff(omega)
    
    # Arc length integral
    arc_length = np.sum(np.sqrt(1 + dmag_domega**2) * np.diff(omega))
    
    # SPARC is negative of arc length
    sparc = -arc_length
    
    return sparc


def local_maxima_count(trajectory, frame_rate, threshold_ratio=0.10):
    """
    Count number of local maxima (peaks) in speed profile.
    
    Smoother movements have fewer submovements and corrections, resulting in fewer
    peaks in the speed profile. This metric counts peaks above a threshold.
    
    Parameters
    ----------
    trajectory : ndarray, shape (n_frames, 3)
        Position trajectory [x, y, z] in spatial units
    frame_rate : float
        Sampling frequency in Hz
    threshold_ratio : float, default=0.10
        Peak detection threshold as ratio of maximum speed (0-1)
    
    Returns
    -------
    n_peaks : int
        Number of peaks in speed profile. Fewer peaks = smoother movement.
    
    Notes
    -----
    - Formula: Count peaks in ||v(t)|| where v(t) > threshold
    - Uses scipy.signal.find_peaks for peak detection
    - Threshold prevents detection of noise-level fluctuations
    - Fewer peaks indicate more ballistic, less corrective movements
    
    References
    ----------
    Rohrer, B., Fasoli, S., Krebs, H. I., Hughes, R., Volpe, B., Frontera, W. R.,
    & Hogan, N. (2002). Movement smoothness changes during stroke recovery.
    Stroke.
    """
    # Compute speed profile
    velocity = np.diff(trajectory, axis=0) * frame_rate
    speed = np.linalg.norm(velocity, axis=1)
    
    # Set threshold for peak detection
    threshold = threshold_ratio * np.max(speed)
    
    # Find peaks above threshold
    peaks, _ = signal.find_peaks(speed, height=threshold)
    
    n_peaks = len(peaks)
    
    return n_peaks


# ============================================================================
# POSTURAL CONTINUITY METRICS
# ============================================================================

def body_posture_rate(positions, frame_rate):
    """
    Compute Body Posture Rate (BPR) - postural continuity metric.
    
    BPR measures the rate of change in whole-body joint configurations. Higher BPR
    indicates more frequent postural adjustments (less smooth). Lower values suggest
    more continuous, flowing postural transitions.
    
    Parameters
    ----------
    positions : ndarray, shape (n_frames, n_joints, 3)
        Array of joint positions over time
    frame_rate : float
        Sampling frequency in Hz
    
    Returns
    -------
    bpr : float
        Body Posture Rate in units per second. Lower values indicate
        smoother postural transitions.
    
    Notes
    -----
    BPR is computed as:
    BPR = BPV/T = (1/n·T) · Σᵢ Σⱼ ||pᵢⱼ - p̄ⱼ||
    
    where:
    - n = number of postures (frames)
    - J = number of joints
    - pᵢⱼ = position of joint j at posture i
    - p̄ⱼ = mean position of joint j across all postures
    - T = movement duration
    
    References
    ----------
    Vatavu, R.-D. (2017). Beyond features for recognition: human-readable
    measures to understand users whole-body gesture performance.
    International Journal of Human-Computer Interaction.
    """
    n_frames, n_joints, n_dims = positions.shape
    
    # Compute mean position for each joint
    mean_positions = np.mean(positions, axis=0)  # shape: (n_joints, 3)
    
    # Compute Body Posture Variability (BPV)
    deviations = positions - mean_positions  # broadcasting
    distances = np.linalg.norm(deviations, axis=2)  # shape: (n_frames, n_joints)
    bpv = np.sum(distances) / n_frames
    
    # Compute duration
    duration = (n_frames - 1) / frame_rate
    
    # BPR = BPV / T
    bpr = bpv / duration if duration > 0 else 0.0
    
    return bpr


# ============================================================================
# SPATIAL REGULARITY METRICS
# ============================================================================

def curvature_variability(trajectory, frame_rate):
    """
    Compute variability of path curvature.
    
    Curvature variability measures how much the path curvature changes along the
    trajectory. Lower values indicate more regular, predictable paths with consistent
    turning rates.
    
    Parameters
    ----------
    trajectory : ndarray, shape (n_frames, 3)
        Position trajectory [x, y, z] in spatial units
    frame_rate : float
        Sampling frequency in Hz
    
    Returns
    -------
    cv : float
        Curvature variability (standard deviation of curvature values)
    
    Notes
    -----
    Curvature κ at each point is computed as:
    κ = ||v × a|| / ||v||³
    
    where:
    - v = velocity vector (first derivative of position)
    - a = acceleration vector (second derivative of position)
    - × denotes cross product
    
    Lower CV values indicate more regular, predictable paths.
    
    References
    ----------
    Charles, S. K., & Hogan, N. (2010). The curvature and variability of
    wrist and arm movements. Experimental Brain Research.
    """
    if len(trajectory) < 3:
        return np.nan
    
    dt = 1.0 / frame_rate
    
    # Compute velocity (first derivative)
    velocity = np.diff(trajectory, axis=0) / dt
    
    # Compute acceleration (second derivative)
    acceleration = np.diff(velocity, axis=0) / dt
    
    # For curvature calculation, we need matching dimensions
    velocity_matched = velocity[:-1]
    
    # Compute curvature at each point: κ = ||v × a|| / ||v||³
    curvatures = []
    for i in range(len(acceleration)):
        v = velocity_matched[i]
        a = acceleration[i]
        
        # Cross product magnitude
        cross_prod = np.cross(v, a)
        cross_magnitude = np.linalg.norm(cross_prod)
        
        # Velocity magnitude cubed
        v_magnitude = np.linalg.norm(v)
        
        if v_magnitude > 1e-6:  # Avoid division by zero
            kappa = cross_magnitude / (v_magnitude ** 3)
            curvatures.append(kappa)
    
    if len(curvatures) == 0:
        return np.nan
    
    # Variability = standard deviation of curvatures
    cv = np.std(curvatures)
    
    return cv


def tortuosity(trajectory):
    """
    Compute path tortuosity - ratio of path length to straight-line distance.
    
    Quantifies how indirect the movement path is. A value of 1.0 indicates
    a perfectly straight path, while higher values indicate more circuitous routes.
    
    Parameters
    ----------
    trajectory : ndarray, shape (n_frames, 3)
        Position trajectory [x, y, z] in spatial units
    
    Returns
    -------
    tort : float
        Tortuosity ratio. Values ≥ 1.0, with 1.0 = perfectly straight path.
    
    Notes
    -----
    Formula: Tortuosity = Lpath / Lstraight
    
    where:
    - Lpath = total path length (sum of step distances)
    - Lstraight = Euclidean distance from start to end
    
    Lower values (closer to 1.0) indicate more direct, efficient movements.
    
    References
    ----------
    Schneiberg, S., Sveistrup, H., McFadyen, B., McKinley, P., & Levin, M. F. (2002).
    The development of coordination for reach-to-grasp movements in children.
    Experimental Brain Research.

    Grace, N., Enticott, P. G., Johnson, B. P., & Rinehart, N. J. (2017).
    Do handwriting difficulties correlate with core symptomology, motor proficiency
    and attentional behaviours? Journal of Autism and Developmental Disorders.
    """
    # Compute path length
    displacements = np.diff(trajectory, axis=0)
    path_length = np.sum(np.linalg.norm(displacements, axis=1))
    
    # Compute straight-line distance (start to end)
    straight_line = np.linalg.norm(trajectory[-1] - trajectory[0])
    
    # Tortuosity ratio
    if straight_line > 0:
        tort = path_length / straight_line
    else:
        tort = np.inf  # Undefined for stationary movement
    
    return tort


# ============================================================================
# RHYTHMIC CONSISTENCY METRICS
# ============================================================================

def harmonic_ratio(trajectory, frame_rate):
    """
    Compute harmonic ratio for rhythmic movements.
    
    The harmonic ratio quantifies movement rhythmicity by comparing the power in
    harmonic frequencies to non-harmonic frequencies.
    Higher values indicate more rhythmic, consistent cyclic movements.
    
    Parameters
    ----------
    trajectory : ndarray, shape (n_frames, 3)
        Position trajectory [x, y, z] in spatial units
    frame_rate : float
        Sampling frequency in Hz
    
    Returns
    -------
    hr : float
        Harmonic ratio. Higher values indicate more rhythmic/symmetric movement.
    
    Notes
    -----
    Formula from Table I: HR = Σ(power at ODD harmonics) / Σ(power at EVEN harmonics).
    
    (Note: This implementation follows the paper's definition explicitly, where A_k are 
    Fourier coefficients. Note that stride frequency is typically fundamental).
    
    References
    ----------
    Bellanca, J. L., Lowry, K. A., Vanswearingen, J. M., Brach, J. S., & Redfern, M. S. (2013).
    Harmonic ratios: a quantitative measure of gait smoothness and rhythmicity.
    Gait & Posture.
    """
    if len(trajectory) < 10:
        return np.nan
    
    # Compute velocity magnitude (speed)
    velocity = np.diff(trajectory, axis=0) * frame_rate
    speed = np.linalg.norm(velocity, axis=1)
    
    # Remove DC component (mean)
    speed_centered = speed - np.mean(speed)
    
    # Apply FFT
    n = len(speed_centered)
    fft_vals = np.fft.rfft(speed_centered)
    power = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0/frame_rate)
    
    # Find fundamental frequency (dominant peak, excluding DC)
    if len(power) < 2:
        return np.nan
    
    # Exclude DC component (index 0)
    dominant_idx = np.argmax(power[1:]) + 1
    fundamental_freq = freqs[dominant_idx]
    
    if fundamental_freq < 0.01:  # Too low frequency
        return np.nan
    
    # Define how many harmonics to include
    n_harmonics = 10
    
    # Calculate power at even harmonics (2f0, 4f0, 6f0, ...)
    even_power = 0.0
    for k in range(2, n_harmonics + 1, 2):
        harmonic_freq = k * fundamental_freq
        idx = np.argmin(np.abs(freqs - harmonic_freq))
        if idx < len(power):
            even_power += power[idx]
    
    # Calculate power at odd harmonics (f0, 3f0, 5f0, ...)
    odd_power = 0.0
    for k in range(1, n_harmonics + 1, 2):
        harmonic_freq = k * fundamental_freq
        idx = np.argmin(np.abs(freqs - harmonic_freq))
        if idx < len(power):
            odd_power += power[idx]
    
    # Compute harmonic ratio according to Table I (Odd / Even)
    if even_power > 0:
        hr = odd_power / even_power
    else:
        hr = np.nan
    
    return hr


def velocity_autocorrelation_decay(trajectory, frame_rate, max_lag=None):
    """
    Compute velocity autocorrelation decay time (VAD).
    
    The velocity autocorrelation decay time quantifies how quickly the predictability
    of movement diminishes over time. Shorter decay time indicates less predictable, more
    variable movements. Longer decay time indicates more rhythmic, consistent timing.
    
    Parameters
    ----------
    trajectory : ndarray, shape (n_frames, 3)
        Position trajectory [x, y, z] in spatial units
    frame_rate : float
        Sampling frequency in Hz
    max_lag : int, optional
        Maximum lag for autocorrelation computation (in frames).
        If None, uses min(len(trajectory)//4, int(frame_rate)).
    
    Returns
    -------
    vad : float
        Velocity autocorrelation decay time. The lag at which autocorrelation
        drops to 1/e (approx 0.368) of its initial value, expressed in seconds.
        Lower values = faster decay = less rhythmic movement.
    
    Notes
    -----
    - Uses autocorrelation decay as defined in Table I.
    - VAD(tau_d) = e^-1 * VAD(0).
    
    References
    ----------
    Kadaba, M. P., Ramakrishnan, H. K., Wootten, M. E., Gainey, J.,
    Gorton, G., & Cochran, G. V. (1990). Repeatability of kinematic, kinetic,
    and electromyographic data in normal adult gait.
    Journal of Orthopaedic Research.
    """
    if len(trajectory) < 10:
        return np.nan
    
    # Compute velocity magnitude (speed)
    velocity = np.diff(trajectory, axis=0) * frame_rate
    speed = np.linalg.norm(velocity, axis=1)
    
    # Set max_lag if not provided
    if max_lag is None:
        max_lag = min(len(speed) // 4, int(frame_rate))
    
    max_lag = min(max_lag, len(speed) - 1)
    
    if max_lag < 2:
        return np.nan
    
    # Center the signal (subtract mean)
    speed_centered = speed - np.mean(speed)
    
    # Compute autocorrelation function
    acf = np.zeros(max_lag + 1)
    variance = np.sum(speed_centered ** 2)
    
    if variance < 1e-10:
        return np.nan
    
    for lag in range(max_lag + 1):
        if lag < len(speed_centered):
            acf[lag] = np.sum(speed_centered[:-lag or None] * speed_centered[lag:]) / variance
    
    # Find decay time: lag where ACF drops to 1/e ~ 0.368
    target_value = 1.0 / np.e
    
    # Find first crossing of threshold
    crossings = np.where(acf < target_value)[0]
    
    if len(crossings) > 0:
        decay_lag = crossings[0]
    else:
        # If never crosses, fit exponential to find decay constant
        positive_acf = acf[acf > 0]
        if len(positive_acf) > 2:
            lags = np.arange(len(positive_acf))
            log_acf = np.log(positive_acf)
            coeffs = np.polyfit(lags, log_acf, 1)
            if coeffs[0] < 0:  # Ensure decaying
                tau = -1.0 / coeffs[0]  # Decay constant in frames
                decay_lag = tau
            else:
                decay_lag = max_lag
        else:
            decay_lag = max_lag
    
    # Convert from frames to seconds
    decay_time = decay_lag / frame_rate
    
    return decay_time