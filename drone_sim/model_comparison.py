import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import signal, stats
import warnings
warnings.filterwarnings('ignore')

# ========== YOUR EXISTING WIND GENERATORS ==========
class OUWindGenerator:
    def __init__(self, dt=0.01):
        self.dt = dt
        self.wind = np.zeros(3)
        self.wind_history = []
        
    def update(self):
        theta = 0.15
        sigma_h = 3.7
        sigma_v = 0.9
        mu_h = 0
        mu_v = 0
        
        self.wind[0] += theta * (mu_h - self.wind[0]) * self.dt + \
                       sigma_h * np.sqrt(self.dt) * np.random.randn()
        self.wind[1] += theta * (mu_h - self.wind[1]) * self.dt + \
                       sigma_h * np.sqrt(self.dt) * np.random.randn()
        self.wind[2] += theta * (mu_v - self.wind[2]) * self.dt + \
                       sigma_v * np.sqrt(self.dt) * np.random.randn()
        
        self.wind_history.append(self.wind.copy())
        return self.wind

class DrydenWindGenerator:
    def __init__(self, dt=0.01):
        self.dt = dt
        self.wind = np.zeros(3)
        self.wind_history = []
        
        self.sigma_u = 1.2
        self.sigma_v = 1.2
        self.sigma_w = 0.1
        
        self.tau_u = 2.0
        self.tau_v = 2.0
        self.tau_w = 4.0
        
        self.u_filter = 0.0
        self.v_filter = 0.0
        self.w_filter = 0.0
        
    def update(self):
        noise_u = np.random.randn()
        noise_v = np.random.randn()
        noise_w = np.random.randn()
        
        self.u_filter += (-self.u_filter / self.tau_u + 
                         self.sigma_u * np.sqrt(2/(self.tau_u * self.dt)) * noise_u) * self.dt
        self.v_filter += (-self.v_filter / self.tau_v + 
                         self.sigma_v * np.sqrt(2/(self.tau_v * self.dt)) * noise_v) * self.dt
        self.w_filter += (-self.w_filter / self.tau_w + 
                         self.sigma_w * np.sqrt(2/(self.tau_w * self.dt)) * noise_w) * self.dt
        
        self.wind[0] = self.u_filter
        self.wind[1] = self.v_filter
        self.wind[2] = self.w_filter * 0.5
        
        if np.random.rand() < 0.003:
            gust_magnitude = np.random.uniform(1.0, 2.5)
            gust_direction = np.random.uniform(0, 2*np.pi)
            self.wind[0] += gust_magnitude * np.cos(gust_direction)
            self.wind[1] += gust_magnitude * np.sin(gust_direction)
        
        self.wind_history.append(self.wind.copy())
        return self.wind

# ========== QUADCOPTER WITH COMPREHENSIVE METRICS ==========
class QuadcopterComprehensive:
    def __init__(self, dt=0.01):
        # Physical properties (using Dryden model parameters)
        self.mass = 0.5
        self.weight = self.mass * 9.81
        self.dt = dt
        self.state = np.zeros(8)
        self.max_thrust = 3.0
        self.hover_thrust = self.weight / 4
        self.motor_thrusts = np.array([self.hover_thrust] * 4)
        
        # Control gains
        self.kp_pos = np.array([12.0, 12.0, 30.0])
        self.kd_pos = np.array([6.0, 6.0, 15.0])
        self.ki_z = 0.8
        self.integral_z = 0
        
        # Storage for comprehensive metrics
        self.positions = []
        self.velocities = []
        self.angles = []
        self.angular_rates = []
        self.motor_history = []
        self.forces = []
        self.accelerations = []
        self.wind_experienced = []  # Store actual wind experienced
        
    def update(self, wind):
        pos = self.state[0:3]
        vel = self.state[3:6]
        pitch = self.state[6]
        roll = self.state[7]
        
        error = -pos
        desired_acc = self.kp_pos * error + self.kd_pos * (-vel)
        
        self.integral_z += error[2] * self.dt
        self.integral_z = np.clip(self.integral_z, -2, 2)
        desired_acc[2] += self.ki_z * self.integral_z
        
        total_force_needed = self.mass * desired_acc
        total_force_needed[2] += self.weight
        total_force_needed += -wind * 0.5
        
        thrust_mag_needed = np.linalg.norm(total_force_needed)
        
        if thrust_mag_needed > 0.01:
            thrust_dir = total_force_needed / thrust_mag_needed
            fx_ratio = np.clip(thrust_dir[0], -0.9, 0.9)
            fy_ratio = np.clip(thrust_dir[1], -0.9, 0.9)
            desired_pitch = np.arcsin(fx_ratio)
            desired_roll = -np.arcsin(fy_ratio)
        else:
            desired_pitch = 0
            desired_roll = 0
            thrust_mag_needed = self.weight
        
        max_angle = np.radians(45)
        desired_pitch = np.clip(desired_pitch, -max_angle, max_angle)
        desired_roll = np.clip(desired_roll, -max_angle, max_angle)
        
        angle_tau = 0.1
        self.state[6] += (desired_pitch - pitch) * self.dt / angle_tau
        self.state[7] += (desired_roll - roll) * self.dt / angle_tau
        
        pitch = self.state[6]
        roll = self.state[7]
        
        base_thrust = thrust_mag_needed / 4
        pitch_adj = pitch * 0.8
        roll_adj = roll * 0.8
        
        self.motor_thrusts = np.array([
            base_thrust - roll_adj + pitch_adj,
            base_thrust + roll_adj + pitch_adj,
            base_thrust - roll_adj - pitch_adj,
            base_thrust + roll_adj - pitch_adj
        ])
        
        self.motor_thrusts = np.maximum(self.motor_thrusts, 0.1)
        
        total_thrust = np.sum(self.motor_thrusts)
        thrust_x = np.sin(pitch) * total_thrust
        thrust_y = -np.sin(roll) * total_thrust
        thrust_z = np.cos(pitch) * np.cos(roll) * total_thrust
        thrust_force = np.array([thrust_x, thrust_y, thrust_z])
        
        drag_coeff = 0.25
        drag_force = -drag_coeff * (vel - wind)
        
        gravity_force = np.array([0, 0, -self.weight])
        total_force = thrust_force + drag_force + gravity_force
        acceleration = total_force / self.mass
        
        self.state[3:6] += acceleration * self.dt
        self.state[0:3] += self.state[3:6] * self.dt
        self.state[0:3] = np.clip(self.state[0:3], -10, 10)
        
        # Store comprehensive data
        self.positions.append(pos.copy())
        self.velocities.append(vel.copy())
        self.angles.append([pitch, roll])
        self.motor_history.append(self.motor_thrusts.copy())
        self.forces.append(total_force.copy())
        self.accelerations.append(acceleration.copy())
        self.wind_experienced.append(wind.copy())  # Store wind
        
        # Calculate angular rates
        if len(self.angles) > 1:
            prev_angles = self.angles[-2]
            current_angles = self.angles[-1]
            angular_rate = [(current_angles[0] - prev_angles[0]) / self.dt,
                          (current_angles[1] - prev_angles[1]) / self.dt,
                          0]
            self.angular_rates.append(angular_rate)
        else:
            self.angular_rates.append([0, 0, 0])
        
        return self.state

# ========== COMPREHENSIVE METRICS CALCULATION ==========
def calculate_comprehensive_metrics(drone, time_array):
    """Calculate all core metrics from simulation data"""
    positions = np.array(drone.positions)
    velocities = np.array(drone.velocities)
    angles = np.degrees(np.array(drone.angles))
    angular_rates = np.array(drone.angular_rates)
    motors = np.array(drone.motor_history)
    forces = np.array(drone.forces)
    accelerations = np.array(drone.accelerations)
    winds = np.array(drone.wind_experienced)  # Get wind data
    
    dt = drone.dt
    metrics = {}
    
    # Steady-state region (last 70% of simulation)
    steady_start = int(0.3 * len(positions))
    pos_steady = positions[steady_start:]
    vel_steady = velocities[steady_start:]
    ang_steady = angles[steady_start:]
    ang_rate_steady = angular_rates[steady_start:]
    winds_steady = winds[steady_start:]  # Steady-state wind
    
    # Wind metrics
    wind_magnitudes = np.linalg.norm(winds, axis=1)
    wind_magnitudes_steady = np.linalg.norm(winds_steady, axis=1)
    
    metrics['mean_wind_speed'] = np.mean(wind_magnitudes)
    metrics['std_wind_speed'] = np.std(wind_magnitudes)
    metrics['max_wind_speed'] = np.max(wind_magnitudes)
    metrics['wind_speed_variance'] = np.var(wind_magnitudes)
    metrics['steady_wind_mean'] = np.mean(wind_magnitudes_steady)
    metrics['steady_wind_std'] = np.std(wind_magnitudes_steady)
    
    # 1. DISPLACEMENT METRICS
    distances = np.linalg.norm(positions, axis=1)
    distances_steady = np.linalg.norm(pos_steady, axis=1)
    
    metrics['rms_displacement'] = np.sqrt(np.mean(distances**2))
    metrics['max_displacement'] = np.max(distances)
    metrics['mean_displacement'] = np.mean(distances)
    
    # Position variance (steady-state)
    metrics['position_variance_x'] = np.var(pos_steady[:, 0])
    metrics['position_variance_y'] = np.var(pos_steady[:, 1])
    metrics['position_variance_z'] = np.var(pos_steady[:, 2])
    metrics['position_variance_total'] = np.mean([metrics['position_variance_x'], 
                                                 metrics['position_variance_y'], 
                                                 metrics['position_variance_z']])
    
    # 95% confidence ellipsoid
    if len(pos_steady) > 10:
        try:
            cov_matrix = np.cov(pos_steady.T)
            eigenvalues = np.linalg.eigvals(cov_matrix)
            eigenvalues = np.maximum(eigenvalues, 1e-10)
            metrics['confidence_ellipsoid_volume'] = np.sqrt(np.prod(eigenvalues)) * 5.991
        except:
            metrics['confidence_ellipsoid_volume'] = 0.0
    else:
        metrics['confidence_ellipsoid_volume'] = 0.0
    
    # Settling radius
    if len(distances_steady) > 10:
        metrics['settling_radius'] = np.percentile(distances_steady, 95)
    else:
        metrics['settling_radius'] = 0.0
    
    # 2. VELOCITY AND ACCELERATION METRICS
    metrics['velocity_variance_x'] = np.var(vel_steady[:, 0])
    metrics['velocity_variance_y'] = np.var(vel_steady[:, 1])
    metrics['velocity_variance_z'] = np.var(vel_steady[:, 2])
    metrics['velocity_variance_total'] = np.mean([metrics['velocity_variance_x'], 
                                                 metrics['velocity_variance_y'], 
                                                 metrics['velocity_variance_z']])
    
    vel_magnitudes = np.linalg.norm(vel_steady, axis=1)
    metrics['rms_velocity'] = np.sqrt(np.mean(vel_magnitudes**2))
    
    acc_magnitudes = np.linalg.norm(accelerations[steady_start:], axis=1)
    metrics['rms_acceleration'] = np.sqrt(np.mean(acc_magnitudes**2))
    metrics['peak_acceleration'] = np.percentile(acc_magnitudes, 99)
    
    # 3. ATTITUDE (RPY) METRICS
    metrics['pitch_variance'] = np.var(ang_steady[:, 0])
    metrics['roll_variance'] = np.var(ang_steady[:, 1])
    
    tilt_angles = np.sqrt(ang_steady[:, 0]**2 + ang_steady[:, 1]**2)
    metrics['rms_tilt_angle'] = np.sqrt(np.mean(tilt_angles**2))
    metrics['max_tilt_angle'] = np.max(tilt_angles)
    
    angular_rate_magnitudes = np.linalg.norm(ang_rate_steady, axis=1)
    metrics['angular_rate_rms'] = np.sqrt(np.mean(angular_rate_magnitudes**2))
    
    # 4. PATH METRICS
    if len(positions) > 2:
        displacements = np.diff(positions, axis=0)
        step_lengths = np.linalg.norm(displacements, axis=1)
        metrics['path_length'] = np.sum(step_lengths)
        
        straight_line = np.linalg.norm(positions[-1] - positions[0])
        if straight_line > 0.01:
            metrics['excessive_path_length'] = metrics['path_length'] / straight_line
        else:
            metrics['excessive_path_length'] = 1.0
        
        curvatures = []
        for i in range(1, len(positions)-1):
            v1 = positions[i] - positions[i-1]
            v2 = positions[i+1] - positions[i]
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            
            if norm_v1 > 1e-5 and norm_v2 > 1e-5:
                cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                curvatures.append(angle / (dt * 2))
        
        if curvatures:
            metrics['mean_curvature'] = np.mean(curvatures)
            metrics['turning_rate_variance'] = np.var(curvatures)
        else:
            metrics['mean_curvature'] = 0.0
            metrics['turning_rate_variance'] = 0.0
    else:
        metrics['path_length'] = 0.0
        metrics['excessive_path_length'] = 1.0
        metrics['mean_curvature'] = 0.0
        metrics['turning_rate_variance'] = 0.0
    
    # 5. MOTOR THRUST VARIANCE
    metrics['motor_thrust_variance'] = np.var(motors[steady_start:])
    
    # 6. TEMPORAL METRICS
    if len(vel_steady) > 10:
        vel_x = vel_steady[:, 0] - np.mean(vel_steady[:, 0])
        autocorr = np.correlate(vel_x, vel_x, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]
        
        threshold = 1/np.e
        valid_indices = np.where(autocorr < threshold)[0]
        if len(valid_indices) > 0:
            metrics['correlation_time'] = valid_indices[0] * dt
        else:
            metrics['correlation_time'] = len(autocorr) * dt
    else:
        metrics['correlation_time'] = 0.0
    
    # Recovery time after gust
    if len(distances) > 10:
        std_steady = np.std(distances_steady)
        recovery_threshold = 2 * std_steady
        
        max_idx = np.argmax(distances)
        recovery_indices = np.where(distances[max_idx:] < recovery_threshold)[0]
        
        if len(recovery_indices) > 0:
            metrics['recovery_time'] = recovery_indices[0] * dt
        else:
            metrics['recovery_time'] = (len(distances) - max_idx) * dt
    else:
        metrics['recovery_time'] = 0.0
    
    return metrics

# ========== SINGLE TRIAL SIMULATION ==========
def run_simulation(wind_type='OU', dt=0.01, duration=30):
    """Run a single simulation with specified wind type"""
    time_array = np.arange(0, duration, dt)
    
    if wind_type == 'OU':
        wind_gen = OUWindGenerator(dt)
    else:  # Dryden
        wind_gen = DrydenWindGenerator(dt)
    
    drone = QuadcopterComprehensive(dt)
    
    # Run simulation
    for t in time_array:
        wind = wind_gen.update()
        drone.update(wind)
    
    # Calculate all metrics
    metrics = calculate_comprehensive_metrics(drone, time_array)
    
    return drone, wind_gen, metrics

# ========== WIND SPEED ANALYSIS FUNCTIONS ==========
def analyze_wind_speed_difference(ou_wind_gen, dryden_wind_gen, time_array):
    """Analyze the difference in wind speed between models"""
    ou_wind_history = np.array(ou_wind_gen.wind_history)
    dryden_wind_history = np.array(dryden_wind_gen.wind_history)
    
    # Convert to km/h for better readability
    ou_wind_kmh = ou_wind_history * 3.6
    dryden_wind_kmh = dryden_wind_history * 3.6
    
    # Calculate wind magnitudes
    ou_wind_magnitude = np.linalg.norm(ou_wind_kmh, axis=1)
    dryden_wind_magnitude = np.linalg.norm(dryden_wind_kmh, axis=1)
    
    # Calculate difference
    wind_speed_difference = dryden_wind_magnitude - ou_wind_magnitude
    
    # Calculate statistics
    stats_dict = {
        'ou_mean': np.mean(ou_wind_magnitude),
        'ou_std': np.std(ou_wind_magnitude),
        'ou_max': np.max(ou_wind_magnitude),
        'dryden_mean': np.mean(dryden_wind_magnitude),
        'dryden_std': np.std(dryden_wind_magnitude),
        'dryden_max': np.max(dryden_wind_magnitude),
        'diff_mean': np.mean(wind_speed_difference),
        'diff_std': np.std(wind_speed_difference),
        'diff_max': np.max(np.abs(wind_speed_difference)),
        'correlation': np.corrcoef(ou_wind_magnitude, dryden_wind_magnitude)[0, 1]
    }
    
    return ou_wind_magnitude, dryden_wind_magnitude, wind_speed_difference, stats_dict

def plot_wind_speed_comparison(ou_data, dryden_data):
    """Create comprehensive wind speed comparison plots"""
    ou_drone, ou_wind, ou_metrics = ou_data
    dryden_drone, dryden_wind, dryden_metrics = dryden_data
    
    time_array = np.arange(0, 30, 0.01)
    
    # Analyze wind speed differences
    ou_wind_mag, dryden_wind_mag, wind_diff, wind_stats = analyze_wind_speed_difference(
        ou_wind, dryden_wind, time_array
    )
    
    # Create figure with 2x2 layout for wind analysis
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Wind Speed Time Series Comparison
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(time_array, ou_wind_mag, 'r-', alpha=0.7, linewidth=1.5, label='OU Wind Model')
    ax1.plot(time_array, dryden_wind_mag, 'b-', alpha=0.7, linewidth=1.5, label='Dryden Wind Model')
    ax1.fill_between(time_array, 0, ou_wind_mag, color='r', alpha=0.2)
    ax1.fill_between(time_array, 0, dryden_wind_mag, color='b', alpha=0.2)
    
    # Add mean lines
    ax1.axhline(y=wind_stats['ou_mean'], color='r', linestyle='--', alpha=0.5, 
               linewidth=1, label=f"OU Mean: {wind_stats['ou_mean']:.1f} km/h")
    ax1.axhline(y=wind_stats['dryden_mean'], color='b', linestyle='--', alpha=0.5,
               linewidth=1, label=f"Dryden Mean: {wind_stats['dryden_mean']:.1f} km/h")
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Wind Speed (km/h)')
    ax1.set_title('WIND SPEED COMPARISON: OU vs DRYDEN MODELS')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_ylim([0, max(wind_stats['ou_max'], wind_stats['dryden_max']) * 1.1])
    
    # 2. Wind Speed Difference Plot
    ax2 = plt.subplot(2, 2, 2)
    
    # Color code by difference sign
    positive_diff = wind_diff > 0
    negative_diff = wind_diff < 0
    
    ax2.fill_between(time_array[positive_diff], 0, wind_diff[positive_diff], 
                     color='b', alpha=0.5, label='Dryden > OU')
    ax2.fill_between(time_array[negative_diff], wind_diff[negative_diff], 0,
                     color='r', alpha=0.5, label='OU > Dryden')
    
    # Plot zero line
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.5, linewidth=0.5)
    
    # Plot mean difference
    ax2.axhline(y=wind_stats['diff_mean'], color='g', linestyle='--', 
               alpha=0.7, linewidth=1.5, 
               label=f'Mean Difference: {wind_stats["diff_mean"]:.2f} km/h')
    
    # Add ±1 std bands
    ax2.axhspan(wind_stats['diff_mean'] - wind_stats['diff_std'],
               wind_stats['diff_mean'] + wind_stats['diff_std'],
               alpha=0.1, color='g', label='±1 Std Dev')
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Wind Speed Difference (km/h)\n(Dryden - OU)')
    ax2.set_title('WIND SPEED DIFFERENCE: Dryden Minus OU')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.set_ylim([-wind_stats['diff_max']*1.1, wind_stats['diff_max']*1.1])
    
    # 3. Wind Speed Distribution Comparison
    ax3 = plt.subplot(2, 2, 3)
    
    # Histogram with KDE
    bins = np.linspace(0, max(wind_stats['ou_max'], wind_stats['dryden_max']), 50)
    
    ax3.hist(ou_wind_mag, bins=bins, alpha=0.5, color='r', 
             density=True, label='OU Model', edgecolor='darkred')
    ax3.hist(dryden_wind_mag, bins=bins, alpha=0.5, color='b', 
             density=True, label='Dryden Model', edgecolor='darkblue')
    
    # Add vertical lines for means
    ax3.axvline(x=wind_stats['ou_mean'], color='r', linestyle='--', 
               linewidth=2, alpha=0.8, label=f"OU Mean: {wind_stats['ou_mean']:.1f}")
    ax3.axvline(x=wind_stats['dryden_mean'], color='b', linestyle='--',
               linewidth=2, alpha=0.8, label=f"Dryden Mean: {wind_stats['dryden_mean']:.1f}")
    
    ax3.set_xlabel('Wind Speed (km/h)')
    ax3.set_ylabel('Probability Density')
    ax3.set_title('WIND SPEED DISTRIBUTION COMPARISON')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right', fontsize=8)
    
    # 4. Wind Speed Statistics Summary
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    # Create summary text
    summary_text = "WIND SPEED STATISTICAL SUMMARY\n"
    summary_text += "=" * 40 + "\n\n"
    
    summary_text += f"OU WIND MODEL:\n"
    summary_text += f"  Mean Speed: {wind_stats['ou_mean']:.2f} km/h\n"
    summary_text += f"  Std Dev: {wind_stats['ou_std']:.2f} km/h\n"
    summary_text += f"  Max Speed: {wind_stats['ou_max']:.1f} km/h\n\n"
    
    summary_text += f"DRYDEN WIND MODEL:\n"
    summary_text += f"  Mean Speed: {wind_stats['dryden_mean']:.2f} km/h\n"
    summary_text += f"  Std Dev: {wind_stats['dryden_std']:.2f} km/h\n"
    summary_text += f"  Max Speed: {wind_stats['dryden_max']:.1f} km/h\n\n"
    
    summary_text += f"DIFFERENCE ANALYSIS:\n"
    summary_text += f"  Mean Diff (Dryden - OU): {wind_stats['diff_mean']:.2f} km/h\n"
    summary_text += f"  Std Dev of Diff: {wind_stats['diff_std']:.2f} km/h\n"
    summary_text += f"  Max |Diff|: {wind_stats['diff_max']:.1f} km/h\n"
    summary_text += f"  Correlation: {wind_stats['correlation']:.3f}\n\n"
    
    # Performance implications
    summary_text += f"PERFORMANCE IMPLICATIONS:\n"
    if wind_stats['dryden_mean'] > wind_stats['ou_mean']:
        diff_pct = ((wind_stats['dryden_mean'] - wind_stats['ou_mean']) / wind_stats['ou_mean']) * 100
        summary_text += f"  • Dryden model has {diff_pct:.1f}% higher mean wind speed\n"
        summary_text += f"  • Expect more control effort with Dryden model\n"
    else:
        diff_pct = ((wind_stats['ou_mean'] - wind_stats['dryden_mean']) / wind_stats['dryden_mean']) * 100
        summary_text += f"  • OU model has {diff_pct:.1f}% higher mean wind speed\n"
        summary_text += f"  • Expect more control effort with OU model\n"
    
    if wind_stats['dryden_std'] > wind_stats['ou_std']:
        summary_text += f"  • Dryden shows more wind speed variability\n"
        summary_text += f"  • May lead to larger position fluctuations\n"
    else:
        summary_text += f"  • OU shows more wind speed variability\n"
        summary_text += f"  • May require faster control responses\n"
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.suptitle('COMPREHENSIVE WIND SPEED ANALYSIS: OU vs DRYDEN MODELS', 
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()
    
    return wind_stats

# ========== MAIN COMPARATIVE ANALYSIS ==========
def main_comparative_analysis(n_trials=500):
    """Run comparative analysis with multiple trials"""
    print(f"\nRunning {n_trials} trials for each wind model...")
    
    ou_all_metrics = []
    dryden_all_metrics = []
    wind_statistics = []
    
    # Last trial data for detailed analysis
    last_ou_data = None
    last_dryden_data = None
    
    # Run OU trials
    print("\n" + "="*50)
    print("Running OU wind model trials...")
    print("="*50)
    
    for i in range(n_trials):
        if i % 50 == 0:
            print(f"  OU Trial {i+1}/{n_trials}")
        
        drone, wind_gen, metrics = run_simulation('OU')
        ou_all_metrics.append(metrics)
        
        if i == n_trials - 1:
            last_ou_data = (drone, wind_gen, metrics)
    
    # Run Dryden trials
    print("\n" + "="*50)
    print("Running Dryden wind model trials...")
    print("="*50)
    
    for i in range(n_trials):
        if i % 50 == 0:
            print(f"  Dryden Trial {i+1}/{n_trials}")
        
        drone, wind_gen, metrics = run_simulation('Dryden')
        dryden_all_metrics.append(metrics)
        
        if i == n_trials - 1:
            last_dryden_data = (drone, wind_gen, metrics)
    
    # Aggregate statistics
    print("\n" + "="*80)
    print(f"AGGREGATE RESULTS (Mean ± Std over {n_trials} trials)")
    print("="*80)
    
    # Core metrics comparison
    core_metrics = [
        'rms_displacement',
        'position_variance_total',
        'velocity_variance_total',
        'rms_tilt_angle',
        'motor_thrust_variance',
        'path_length',
        'correlation_time',
        'recovery_time',
        'mean_wind_speed'  # Added wind speed to comparison
    ]
    
    print(f"\n{'Metric':<30} {'OU Model':<20} {'Dryden Model':<20} {'Difference (%)':<15}")
    print("-" * 85)
    
    for metric in core_metrics:
        ou_vals = [m[metric] for m in ou_all_metrics]
        dryden_vals = [m[metric] for m in dryden_all_metrics]
        
        ou_mean = np.mean(ou_vals)
        ou_std = np.std(ou_vals)
        dryden_mean = np.mean(dryden_vals)
        dryden_std = np.std(dryden_vals)
        
        if ou_mean > 0:
            diff_pct = ((dryden_mean - ou_mean) / ou_mean) * 100
        else:
            diff_pct = 0
        
        print(f"{metric:<30} {ou_mean:.4f} ± {ou_std:.4f}   {dryden_mean:.4f} ± {dryden_std:.4f}   {diff_pct:+.1f}%")
    
    # Plot wind speed comparison from last trial
    print("\n" + "="*80)
    print("WIND SPEED COMPARISON ANALYSIS")
    print("="*80)
    
    wind_stats = plot_wind_speed_comparison(last_ou_data, last_dryden_data)
    
    # Print wind statistics
    print(f"\nWind Speed Statistics (from last trial):")
    print(f"  OU Model: Mean = {wind_stats['ou_mean']:.2f} km/h, Std = {wind_stats['ou_std']:.2f} km/h")
    print(f"  Dryden Model: Mean = {wind_stats['dryden_mean']:.2f} km/h, Std = {wind_stats['dryden_std']:.2f} km/h")
    print(f"  Difference: Mean = {wind_stats['diff_mean']:.2f} km/h, Correlation = {wind_stats['correlation']:.3f}")
    
    # Create comprehensive plots
    plot_comprehensive_comparison(last_ou_data, last_dryden_data, ou_all_metrics, dryden_all_metrics)
    
    # Additional statistical analysis
    statistical_significance_test(ou_all_metrics, dryden_all_metrics)
    
    return ou_all_metrics, dryden_all_metrics, last_ou_data, last_dryden_data, wind_stats

# ========== VISUALIZATION FUNCTIONS ==========
def plot_comprehensive_comparison(ou_data, dryden_data, ou_all_metrics, dryden_all_metrics):
    """Plot comprehensive comparison"""
    ou_drone, ou_wind, ou_metrics = ou_data
    dryden_drone, dryden_wind, dryden_metrics = dryden_data
    
    time_array = np.arange(0, 30, 0.01)
    
    # Create figure with comprehensive layout
    fig = plt.figure(figsize=(20, 15))
    
    # 1. POSITION COMPARISON
    ax1 = plt.subplot(3, 4, 1)
    ou_positions = np.array(ou_drone.positions)
    dryden_positions = np.array(dryden_drone.positions)
    
    ax1.plot(time_array, ou_positions[:, 0], 'r-', alpha=0.7, label='OU X')
    ax1.plot(time_array, ou_positions[:, 1], 'r--', alpha=0.7, label='OU Y')
    ax1.plot(time_array, dryden_positions[:, 0], 'b-', alpha=0.7, label='Dryden X')
    ax1.plot(time_array, dryden_positions[:, 1], 'b--', alpha=0.7, label='Dryden Y')
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position (m)')
    ax1.set_title('POSITION COMPARISON (XY Plane)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)
    ax1.set_ylim([-0.3, 0.3])
    
    # 2. WIND SPEED COMPARISON
    ax2 = plt.subplot(3, 4, 2)
    ou_wind_history = np.array(ou_wind.wind_history) * 3.6
    dryden_wind_history = np.array(dryden_wind.wind_history) * 3.6
    
    ou_wind_mag = np.linalg.norm(ou_wind_history, axis=1)
    dryden_wind_mag = np.linalg.norm(dryden_wind_history, axis=1)
    
    ax2.plot(time_array, ou_wind_mag, 'r-', alpha=0.7, linewidth=1, label='OU')
    ax2.plot(time_array, dryden_wind_mag, 'b-', alpha=0.7, linewidth=1, label='Dryden')
    ax2.fill_between(time_array, 0, ou_wind_mag, color='r', alpha=0.2)
    ax2.fill_between(time_array, 0, dryden_wind_mag, color='b', alpha=0.2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Wind Speed (km/h)')
    ax2.set_title('ABSOLUTE WIND SPEED COMPARISON')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. TILT ANGLE COMPARISON
    ax3 = plt.subplot(3, 4, 3)
    ou_angles = np.degrees(np.array(ou_drone.angles))
    dryden_angles = np.degrees(np.array(dryden_drone.angles))
    
    ou_tilt = np.sqrt(ou_angles[:, 0]**2 + ou_angles[:, 1]**2)
    dryden_tilt = np.sqrt(dryden_angles[:, 0]**2 + dryden_angles[:, 1]**2)
    
    ax3.plot(time_array, ou_tilt, 'r-', alpha=0.7, label='OU')
    ax3.plot(time_array, dryden_tilt, 'b-', alpha=0.7, label='Dryden')
    ax3.axhline(y=45, color='k', linestyle='--', alpha=0.5, label='45° limit')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Tilt Angle (deg)')
    ax3.set_title('TILT ANGLE COMPARISON')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_ylim([0, 50])
    
    # 4. POSITION SCATTER (Steady State)
    ax4 = plt.subplot(3, 4, 4)
    steady_start = int(0.3 * len(ou_positions))
    ou_pos_steady = ou_positions[steady_start:, 0:2]
    dryden_pos_steady = dryden_positions[steady_start:, 0:2]
    
    ax4.scatter(ou_pos_steady[:, 0], ou_pos_steady[:, 1], 
               c='r', alpha=0.3, s=10, label='OU', marker='o')
    ax4.scatter(dryden_pos_steady[:, 0], dryden_pos_steady[:, 1], 
               c='b', alpha=0.3, s=10, label='Dryden', marker='^')
    ax4.scatter(0, 0, c='green', s=100, marker='*', label='Target')
    ax4.set_xlabel('X Position (m)')
    ax4.set_ylabel('Y Position (m)')
    ax4.set_title('STEADY-STATE POSITION SCATTER')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=8)
    ax4.set_xlim([-0.2, 0.2])
    ax4.set_ylim([-0.2, 0.2])
    ax4.set_aspect('equal')
    
    # 5. CORE METRICS BAR CHART
    ax5 = plt.subplot(3, 4, 5)
    
    core_metrics_plot = ['rms_displacement', 'position_variance_total', 
                        'velocity_variance_total', 'rms_tilt_angle']
    
    ou_vals = [ou_metrics[m] for m in core_metrics_plot]
    dryden_vals = [dryden_metrics[m] for m in core_metrics_plot]
    
    x = np.arange(len(core_metrics_plot))
    width = 0.35
    
    ax5.bar(x - width/2, ou_vals, width, label='OU', color='r', alpha=0.7)
    ax5.bar(x + width/2, dryden_vals, width, label='Dryden', color='b', alpha=0.7)
    
    ax5.set_xlabel('Metric')
    ax5.set_ylabel('Value')
    ax5.set_title('CORE METRICS COMPARISON (Last Trial)')
    ax5.set_xticks(x)
    ax5.set_xticklabels(['RMS Disp.', 'Pos. Var.', 'Vel. Var.', 'RMS Tilt'], rotation=45)
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. RECOVERY TIME DISTRIBUTION
    ax6 = plt.subplot(3, 4, 6)
    
    ou_recovery = [m['recovery_time'] for m in ou_all_metrics]
    dryden_recovery = [m['recovery_time'] for m in dryden_all_metrics]
    
    bins = np.linspace(0, max(max(ou_recovery), max(dryden_recovery)) + 0.5, 30)
    ax6.hist(ou_recovery, bins=bins, alpha=0.5, color='r', label='OU', edgecolor='darkred')
    ax6.hist(dryden_recovery, bins=bins, alpha=0.5, color='b', label='Dryden', edgecolor='darkblue')
    
    ax6.set_xlabel('Recovery Time (s)')
    ax6.set_ylabel('Frequency')
    ax6.set_title('RECOVERY TIME DISTRIBUTION')
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 7. CORRELATION TIME COMPARISON
    ax7 = plt.subplot(3, 4, 7)
    
    ou_corr = [m['correlation_time'] for m in ou_all_metrics]
    dryden_corr = [m['correlation_time'] for m in dryden_all_metrics]
    
    box_data = [ou_corr, dryden_corr]
    box_labels = ['OU', 'Dryden']
    
    bp = ax7.boxplot(box_data, labels=box_labels, patch_artist=True)
    
    colors = ['lightcoral', 'lightblue']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax7.set_ylabel('Correlation Time (s)')
    ax7.set_title('VELOCITY CORRELATION TIME')
    ax7.grid(True, alpha=0.3, axis='y')
    
    # 8. 3D TRAJECTORY COMPARISON
    ax8 = plt.subplot(3, 4, 8, projection='3d')
    ax8.plot(ou_positions[:, 0], ou_positions[:, 1], ou_positions[:, 2], 
             'r-', alpha=0.7, linewidth=1, label='OU')
    ax8.plot(dryden_positions[:, 0], dryden_positions[:, 1], dryden_positions[:, 2], 
             'b-', alpha=0.7, linewidth=1, label='Dryden')
    ax8.scatter(0, 0, 0, c='green', s=100, marker='*', label='Target')
    ax8.set_xlabel('X (m)')
    ax8.set_ylabel('Y (m)')
    ax8.set_zlabel('Z (m)')
    ax8.set_title('3D TRAJECTORY COMPARISON')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    ax8.set_xlim([-0.5, 0.5])
    ax8.set_ylim([-0.5, 0.5])
    ax8.set_zlim([-0.3, 0.3])
    
    # 9. MOTOR THRUST VARIANCE
    ax9 = plt.subplot(3, 4, 9)
    
    ou_motor_vars = [m['motor_thrust_variance'] for m in ou_all_metrics]
    dryden_motor_vars = [m['motor_thrust_variance'] for m in dryden_all_metrics]
    
    box_data = [ou_motor_vars, dryden_motor_vars]
    bp = ax9.boxplot(box_data, labels=['OU', 'Dryden'], patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax9.set_ylabel('Motor Thrust Variance (N²)')
    ax9.set_title('MOTOR THRUST VARIANCE')
    ax9.grid(True, alpha=0.3, axis='y')
    
    # 10. PATH LENGTH COMPARISON
    ax10 = plt.subplot(3, 4, 10)
    
    ou_path_lengths = [m['path_length'] for m in ou_all_metrics]
    dryden_path_lengths = [m['path_length'] for m in dryden_all_metrics]
    
    ax10.hist(ou_path_lengths, bins=30, alpha=0.5, color='r', label='OU', edgecolor='darkred')
    ax10.hist(dryden_path_lengths, bins=30, alpha=0.5, color='b', label='Dryden', edgecolor='darkblue')
    
    ax10.set_xlabel('Path Length (m)')
    ax10.set_ylabel('Frequency')
    ax10.set_title('PATH LENGTH DISTRIBUTION')
    ax10.legend()
    ax10.grid(True, alpha=0.3, axis='y')
    
    # 11. WIND SPEED DIFFERENCE HISTOGRAM
    ax11 = plt.subplot(3, 4, 11)
    
    ou_wind_speeds = [m['mean_wind_speed'] for m in ou_all_metrics]
    dryden_wind_speeds = [m['mean_wind_speed'] for m in dryden_all_metrics]
    
    wind_differences = np.array(dryden_wind_speeds) - np.array(ou_wind_speeds)
    
    ax11.hist(wind_differences, bins=30, alpha=0.7, color='purple', edgecolor='darkviolet')
    ax11.axvline(x=0, color='k', linestyle='--', linewidth=1, alpha=0.5, label='Zero Difference')
    ax11.axvline(x=np.mean(wind_differences), color='g', linestyle='-', linewidth=2, 
                alpha=0.7, label=f'Mean: {np.mean(wind_differences):.2f} km/h')
    
    ax11.set_xlabel('Wind Speed Difference (km/h)\n(Dryden - OU)')
    ax11.set_ylabel('Frequency')
    ax11.set_title('WIND SPEED DIFFERENCE DISTRIBUTION\n(All 500 Trials)')
    ax11.legend()
    ax11.grid(True, alpha=0.3, axis='y')
    
    # 12. AGGREGATE PERFORMANCE SUMMARY
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')
    
    # Calculate aggregate statistics
    summary_text = "AGGREGATE PERFORMANCE (500 trials)\n"
    summary_text += "=" * 40 + "\n\n"
    
    metrics_summary = [
        ('RMS Displacement', 'rms_displacement'),
        ('Position Variance', 'position_variance_total'),
        ('Velocity Variance', 'velocity_variance_total'),
        ('RMS Tilt Angle', 'rms_tilt_angle'),
        ('Mean Wind Speed', 'mean_wind_speed'),
        ('Recovery Time', 'recovery_time')
    ]
    
    for name, key in metrics_summary:
        ou_vals = [m[key] for m in ou_all_metrics]
        dryden_vals = [m[key] for m in dryden_all_metrics]
        
        ou_mean = np.mean(ou_vals)
        dryden_mean = np.mean(dryden_vals)
        
        if ou_mean > 0:
            improvement = ((ou_mean - dryden_mean) / ou_mean) * 100
        else:
            improvement = 0
        
        summary_text += f"{name}:\n"
        summary_text += f"  OU: {ou_mean:.4f}\n"
        summary_text += f"  Dryden: {dryden_mean:.4f}\n"
        summary_text += f"  Diff: {improvement:+.1f}%\n\n"
    
    ax12.text(0, 1, summary_text, transform=ax12.transAxes, 
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('COMPREHENSIVE COMPARISON: OU vs DRYDEN WIND MODELS', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

# ========== STATISTICAL SIGNIFICANCE TEST ==========
def statistical_significance_test(ou_all_metrics, dryden_all_metrics):
    """Perform statistical significance tests"""
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE TEST (Welch's t-test)")
    print("="*80)
    
    core_metrics = [
        'rms_displacement',
        'position_variance_total',
        'velocity_variance_total',
        'rms_tilt_angle',
        'motor_thrust_variance',
        'correlation_time',
        'recovery_time',
        'mean_wind_speed'
    ]
    
    print(f"\n{'Metric':<30} {'OU Mean':<12} {'Dryden Mean':<12} {'t-statistic':<12} {'p-value':<12} {'Significant?'}")
    print("-" * 90)
    
    results = []
    for metric in core_metrics:
        ou_vals = [m[metric] for m in ou_all_metrics]
        dryden_vals = [m[metric] for m in dryden_all_metrics]
        
        # Perform Welch's t-test
        t_stat, p_value = stats.ttest_ind(ou_vals, dryden_vals, equal_var=False)
        
        ou_mean = np.mean(ou_vals)
        dryden_mean = np.mean(dryden_vals)
        
        significant = "YES" if p_value < 0.05 else "NO"
        
        print(f"{metric:<30} {ou_mean:.4f}      {dryden_mean:.4f}      {t_stat:8.4f}    {p_value:.2e}       {significant}")
        
        results.append({
            'metric': metric,
            'ou_mean': ou_mean,
            'dryden_mean': dryden_mean,
            't_stat': t_stat,
            'p_value': p_value,
            'significant': significant
        })
    
    return results

# ========== MAIN EXECUTION ==========
def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("COMPREHENSIVE QUADCOPTER PERFORMANCE ANALYSIS")
    print("="*80)
    
    print("\nThis analysis will compute all core metrics including:")
    print("1. Wind Speed Analysis: Absolute speed comparison and difference plots")
    print("2. Displacement: RMS, variance, confidence ellipsoid, settling radius")
    print("3. Velocity: Variance, RMS, acceleration RMS, peak acceleration")
    print("4. Attitude: Pitch/roll variance, RMS tilt, angular rate RMS")
    print("5. Path: Path length, excessive path length, curvature, turning rate")
    print("6. Temporal: PSD, autocorrelation, correlation time, recovery time")
    print("7. Motor: Thrust variance")
    
    # Run comparative analysis
    n_trials = 500
    ou_all_metrics, dryden_all_metrics, last_ou, last_dryden, wind_stats = main_comparative_analysis(n_trials)
    
    # Create detailed performance report
    print("\n" + "="*80)
    print("DETAILED PERFORMANCE REPORT")
    print("="*80)
    
    # Calculate overall performance score
    def calculate_performance_score(metrics_list):
        """Calculate overall performance score (lower is better)"""
        weights = {
            'rms_displacement': 0.20,
            'position_variance_total': 0.15,
            'velocity_variance_total': 0.15,
            'rms_tilt_angle': 0.15,
            'motor_thrust_variance': 0.10,
            'recovery_time': 0.10,
            'correlation_time': 0.10,
            'mean_wind_speed': 0.05  # Weight for wind intensity
        }
        
        score = 0
        for metric, weight in weights.items():
            values = [m[metric] for m in metrics_list]
            score += np.mean(values) * weight
        
        return score
    
    ou_score = calculate_performance_score(ou_all_metrics)
    dryden_score = calculate_performance_score(dryden_all_metrics)
    
    print(f"\nOverall Performance Score (lower is better):")
    print(f"  OU Model: {ou_score:.4f}")
    print(f"  Dryden Model: {dryden_score:.4f}")
    
    improvement_pct = ((ou_score - dryden_score) / ou_score) * 100
    print(f"\nPerformance Improvement: {improvement_pct:+.1f}%")
    
    if improvement_pct > 0:
        print("✓ Dryden model shows BETTER overall performance")
    else:
        print("✓ OU model shows BETTER overall performance")
    
    # Wind speed specific analysis
    print("\n" + "="*80)
    print("WIND SPEED SPECIFIC FINDINGS")
    print("="*80)
    
    ou_wind_speeds = [m['mean_wind_speed'] for m in ou_all_metrics]
    dryden_wind_speeds = [m['mean_wind_speed'] for m in dryden_all_metrics]
    
    ou_wind_mean = np.mean(ou_wind_speeds)
    dryden_wind_mean = np.mean(dryden_wind_speeds)
    
    wind_diff_pct = ((dryden_wind_mean - ou_wind_mean) / ou_wind_mean) * 100
    
    print(f"\nWind Speed Statistics (500 trials):")
    print(f"  OU Mean Wind Speed: {ou_wind_mean:.2f} km/h")
    print(f"  Dryden Mean Wind Speed: {dryden_wind_mean:.2f} km/h")
    print(f"  Difference: {wind_diff_pct:+.1f}%")
    
    if wind_diff_pct > 0:
        print(f"\n✓ Dryden model generates {wind_diff_pct:.1f}% stronger winds on average")
        print("  This explains higher control effort in some metrics")
    else:
        print(f"\n✓ OU model generates {abs(wind_diff_pct):.1f}% stronger winds on average")
        print("  This explains higher control effort in some metrics")
    
    # Print recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    print("\nBased on the comprehensive analysis:")
    
    # Check specific metrics for recommendations
    metrics_to_check = [
        ('rms_displacement', 'Position tracking accuracy'),
        ('recovery_time', 'Disturbance rejection'),
        ('motor_thrust_variance', 'Control effort efficiency'),
        ('rms_tilt_angle', 'Stability and passenger comfort'),
        ('mean_wind_speed', 'Wind intensity')
    ]
    
    for metric_key, description in metrics_to_check:
        ou_mean = np.mean([m[metric_key] for m in ou_all_metrics])
        dryden_mean = np.mean([m[metric_key] for m in dryden_all_metrics])
        
        if dryden_mean < ou_mean:
            improvement = ((ou_mean - dryden_mean) / ou_mean) * 100
            print(f"✓ Dryden model shows {improvement:+.1f}% improvement in {description}")
        else:
            improvement = ((dryden_mean - ou_mean) / dryden_mean) * 100
            print(f"✓ OU model shows {improvement:+.1f}% improvement in {description}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()