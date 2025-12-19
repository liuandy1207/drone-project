import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.signal import welch, lti, cont2discrete

print("=" * 80)
print("QUADCOPTER SIMULATION - TRUE SECOND-ORDER DRYDEN TURBULENCE")
print("=" * 80)

# ========== TRUE SECOND-ORDER DRYDEN TURBULENCE GENERATOR ==========
class TrueDrydenWindGenerator:
    def __init__(self, dt=0.01, V=5.0, altitude=10.0, turbulence_level='light'):
        """
        True Dryden turbulence model from MIL-F-8785C
        Uses second-order shaping filters with proper PSD
        """
        self.dt = dt
        self.V = V  # Airspeed (m/s)
        self.altitude = altitude
        self.turbulence_level = turbulence_level
        
        # Set turbulence intensity (sigma) based on level
        if turbulence_level == 'light':
            self.sigma_u = 0.5  # m/s
            self.sigma_v = 0.5  # m/s
            self.sigma_w = 0.1  # m/s
        elif turbulence_level == 'moderate':
            self.sigma_u = 1.0  # m/s
            self.sigma_v = 1.0  # m/s
            self.sigma_w = 0.2  # m/s
        elif turbulence_level == 'severe':
            self.sigma_u = 2.0  # m/s
            self.sigma_v = 2.0  # m/s
            self.sigma_w = 0.4  # m/s
        else:
            self.sigma_u = 0.8  # m/s
            self.sigma_v = 0.8  # m/s
            self.sigma_w = 0.15  # m/s
        
        # Dryden scale lengths (meters) from MIL-F-8785C
        # These depend on altitude
        if altitude < 1000:
            # Low altitude (below 1000 ft / 305 m)
            self.L_u = altitude / (0.177 + 0.000823*altitude)**1.2
            self.L_v = self.L_u / 2  # Lateral scale length is half of longitudinal
            self.L_w = altitude  # Vertical scale length equals altitude
        else:
            # High altitude
            self.L_u = 1750
            self.L_v = self.L_u / 2
            self.L_w = 1750
        
        # Create discrete-time filters for each component
        # Dryden transfer functions:
        # H_u(s) = σ_u * sqrt(2L_u/(πV)) * 1/(1 + L_u*s/V)
        # H_v(s) = σ_v * sqrt(L_v/(πV)) * (1 + sqrt(3)*L_v*s/V)/(1 + L_v*s/V)^2
        # H_w(s) = σ_w * sqrt(L_w/(πV)) * (1 + sqrt(3)*L_w*s/V)/(1 + L_w*s/V)^2
        
        # Pre-calculate common terms
        sqrt_pi_V = np.sqrt(np.pi * V)
        
        # ===== LONGITUDINAL (u-component) - FIRST ORDER =====
        # H_u(s) = σ_u * sqrt(2L_u/(πV)) * 1/(1 + L_u*s/V)
        # State space: A = -V/L_u, B = σ_u * sqrt(2V/(πL_u)), C = 1, D = 0
        
        A_u = np.array([[-self.V / self.L_u]])
        B_u = np.array([[self.sigma_u * np.sqrt(2*self.V/(np.pi*self.L_u))]])
        C_u = np.array([[1.0]])
        D_u = np.array([[0.0]])
        
        # ===== LATERAL (v-component) - SECOND ORDER =====
        # H_v(s) = σ_v * sqrt(L_v/(πV)) * (1 + sqrt(3)*L_v*s/V)/(1 + L_v*s/V)^2
        # Expanded: H_v(s) = σ_v * sqrt(L_v/(πV)) * (sqrt(3)*L_v/V * s + 1) / (L_v^2/V^2 * s^2 + 2L_v/V * s + 1)
        
        L_v_V = self.L_v / self.V
        sigma_v_norm = self.sigma_v * np.sqrt(self.L_v/(np.pi*self.V))
        
        A_v = np.array([[-2/L_v_V, -1/(L_v_V**2)],
                        [1, 0]])
        B_v = np.array([[sigma_v_norm * np.sqrt(3)*L_v_V],
                        [0]])
        C_v = np.array([[1.0, sigma_v_norm]])
        D_v = np.array([[0.0]])
        
        # ===== VERTICAL (w-component) - SECOND ORDER =====
        # Same form as v-component
        L_w_V = self.L_w / self.V
        sigma_w_norm = self.sigma_w * np.sqrt(self.L_w/(np.pi*self.V))
        
        A_w = np.array([[-2/L_w_V, -1/(L_w_V**2)],
                        [1, 0]])
        B_w = np.array([[sigma_w_norm * np.sqrt(3)*L_w_V],
                        [0]])
        C_w = np.array([[1.0, sigma_w_norm]])
        D_w = np.array([[0.0]])
        
        # Convert to discrete-time using zero-order hold (FIXED)
        sys_u_d = cont2discrete((A_u, B_u, C_u, D_u), dt, method='zoh')
        sys_v_d = cont2discrete((A_v, B_v, C_v, D_v), dt, method='zoh')
        sys_w_d = cont2discrete((A_w, B_w, C_w, D_w), dt, method='zoh')
        
        self.Ad_u, self.Bd_u, self.Cd_u, self.Dd_u = sys_u_d[0], sys_u_d[1], sys_u_d[2], sys_u_d[3]
        self.Ad_v, self.Bd_v, self.Cd_v, self.Dd_v = sys_v_d[0], sys_v_d[1], sys_v_d[2], sys_v_d[3]
        self.Ad_w, self.Bd_w, self.Cd_w, self.Dd_w = sys_w_d[0], sys_w_d[1], sys_w_d[2], sys_w_d[3]
        
        # State variables
        self.x_u = np.zeros((1, 1))
        self.x_v = np.zeros((2, 1))
        self.x_w = np.zeros((2, 1))
        
        # Output
        self.wind = np.zeros(3)
        self.wind_history = []
        
        # White noise generator
        self.rng = np.random.default_rng(seed=42)
        
        print(f"\nTRUE DRYDEN MODEL PARAMETERS:")
        print(f"  Airspeed V: {V:.1f} m/s ({V*3.6:.1f} km/h)")
        print(f"  Altitude: {altitude:.1f} m")
        print(f"  Scale lengths: L_u={self.L_u:.1f}m, L_v={self.L_v:.1f}m, L_w={self.L_w:.1f}m")
        print(f"  Turbulence intensities: σ_u={self.sigma_u:.3f} m/s, σ_v={self.sigma_v:.3f} m/s, σ_w={self.sigma_w:.3f} m/s")
        print(f"  Filter orders: X=1st, Y=2nd, Z=2nd")
    
    def update(self):
        # Generate white Gaussian noise (zero mean, unit variance)
        white_u = self.rng.standard_normal()
        white_v = self.rng.standard_normal()
        white_w = self.rng.standard_normal()
        
        # Update longitudinal component (first order)
        self.x_u = self.Ad_u @ self.x_u + self.Bd_u * white_u
        u_turb = float(self.Cd_u @ self.x_u + self.Dd_u * white_u)
        
        # Update lateral component (second order)
        self.x_v = self.Ad_v @ self.x_v + self.Bd_v * white_v
        v_turb = float(self.Cd_v @ self.x_v + self.Dd_v * white_v)
        
        # Update vertical component (second order)
        self.x_w = self.Ad_w @ self.x_w + self.Bd_w * white_w
        w_turb = float(self.Cd_w @ self.x_w + self.Dd_w * white_w)
        
        # Combine components
        self.wind = np.array([u_turb, v_turb, w_turb])
        
        self.wind_history.append(self.wind.copy())
        return self.wind
    
    def theoretical_psd(self, f):
        """Calculate theoretical PSD for verification"""
        omega = 2 * np.pi * f
        
        # Longitudinal PSD (1st order low-pass)
        Phi_u = (self.sigma_u**2 * 2 * self.L_u / (np.pi * self.V)) / (1 + (self.L_u * omega / self.V)**2)
        
        # Lateral PSD (2nd order with numerator dynamics)
        Phi_v = (self.sigma_v**2 * self.L_v / (np.pi * self.V)) * (1 + 3*(self.L_v * omega / self.V)**2) / (1 + (self.L_v * omega / self.V)**2)**2
        
        # Vertical PSD (same form as lateral)
        Phi_w = (self.sigma_w**2 * self.L_w / (np.pi * self.V)) * (1 + 3*(self.L_w * omega / self.V)**2) / (1 + (self.L_w * omega / self.V)**2)**2
        
        return Phi_u, Phi_v, Phi_w

# ========== QUADCOPTER (UNCHANGED) ==========
class Quadcopter:
    def __init__(self, dt=0.01):
        # Physical properties - 0.5 kg (500g)
        self.mass = 0.5                    # mass of the quadcopter (kg) - 500g
        self.weight = self.mass * 9.81    # gravitational force mg (N)

        self.dt = dt                         # simulation timestep (s)

        # State vector: [x, y, z, vx, vy, vz, pitch, roll]
        self.state = np.zeros(8)

        # Motor thrust limits
        self.max_thrust = 3.0           # maximum thrust a motor can output
        self.hover_thrust = self.weight / 4  # thrust needed per motor to hover
        self.motor_thrusts = np.array([self.hover_thrust] * 4)

        # Position controller gains (PD + integral on z)
        self.kp_pos = np.array([12.0, 12.0, 30.0])   # proportional gains (x,y,z)
        self.kd_pos = np.array([6.0, 6.0, 15.0])     # derivative gains (x,y,z)
        self.ki_z = 0.8                            # integral gain for altitude
        self.integral_z = 0                        # accumulated z-error

        # Logging history
        self.positions = []
        self.angles = []
        self.motor_history = []
        self.forces = []

    def update(self, wind):
        # Extract position, velocity, pitch, and roll
        pos = self.state[0:3]
        vel = self.state[3:6]
        pitch = self.state[6]
        roll = self.state[7]

        # Position error (reference point is (0,0,0))
        error = -pos

        # Compute acceleration
        desired_acc = self.kp_pos * error + self.kd_pos * (-vel)

        # Integral control altitude (z-axis), because wind is assumed to predominantly in the x and y directions
        self.integral_z += error[2] * self.dt
        self.integral_z = np.clip(self.integral_z, -2, 2)
        desired_acc[2] += self.ki_z * self.integral_z

        # Convert acceleration into required forces
        total_force_needed = self.mass * desired_acc

        # Force of gravity
        total_force_needed[2] += self.weight

        # Add wind disturbance compensation
        total_force_needed += -wind * 0.5

        # Magnitude of needed thrust 
        thrust_mag_needed = np.linalg.norm(total_force_needed)

        # Calculate desired pitch and roll based on force direction
        if thrust_mag_needed > 0.01:
            thrust_dir = total_force_needed / thrust_mag_needed

            # Horizontal thrust components determine pitch and roll
            fx_ratio = np.clip(thrust_dir[0], -0.9, 0.9)
            fy_ratio = np.clip(thrust_dir[1], -0.9, 0.9)

            desired_pitch = np.arcsin(fx_ratio)
            desired_roll = -np.arcsin(fy_ratio)
        else:
            # If no force required, the angles of pitch and roll are 0
            desired_pitch = 0
            desired_roll = 0
            thrust_mag_needed = self.weight

        # Limit pitch and roll angles to ±45°
        max_angle = np.radians(45)
        desired_pitch = np.clip(desired_pitch, -max_angle, max_angle)
        desired_roll = np.clip(desired_roll, -max_angle, max_angle)

        # First-order angle response (simulates inertia + motor lag)
        angle_tau = 0.1
        self.state[6] += (desired_pitch - pitch) * self.dt / angle_tau
        self.state[7] += (desired_roll - roll) * self.dt / angle_tau

        # Update angles after dynamics
        pitch = self.state[6]
        roll = self.state[7]

        # Compute thrust for 4 motors
        base_thrust = thrust_mag_needed / 4
        pitch_adj = pitch * 0.8
        roll_adj = roll * 0.8

        # Combined thrust from all 4 motors
        self.motor_thrusts = np.array([
            base_thrust - roll_adj + pitch_adj,   # Front-left
            base_thrust + roll_adj + pitch_adj,   # Front-right
            base_thrust - roll_adj - pitch_adj,   # Back-left
            base_thrust + roll_adj - pitch_adj    # Back-right
        ])

        # Prevent motors from going to zero
        self.motor_thrusts = np.maximum(self.motor_thrusts, 0.1)

        # Compute thrust forces
        total_thrust = np.sum(self.motor_thrusts)
        thrust_x = np.sin(pitch) * total_thrust
        thrust_y = -np.sin(roll) * total_thrust
        thrust_z = np.cos(pitch) * np.cos(roll) * total_thrust
        thrust_force = np.array([thrust_x, thrust_y, thrust_z])

        # Parameters for drag force
        drag_coeff = 0.25
        drag_force = -drag_coeff * (vel - wind)

        # Gravity 
        gravity_force = np.array([0, 0, -self.weight])

        # Net force
        total_force = thrust_force + drag_force + gravity_force

        # F = ma 
        acceleration = total_force / self.mass

        # Update velocity with acceleration
        self.state[3:6] += acceleration * self.dt

        # Update position
        self.state[0:3] += self.state[3:6] * self.dt

        # Limit position to within ±10 m
        self.state[0:3] = np.clip(self.state[0:3], -10, 10)

        # Log changes
        self.positions.append(pos.copy())
        self.angles.append([pitch, roll])
        self.motor_history.append(self.motor_thrusts.copy())
        self.forces.append(total_force.copy())

        return self.state
    
# ========== MAIN SIMULATION ==========
def main():
    dt = 0.01
    duration = 30
    time_array = np.arange(0, duration, dt)
    
    # Create TRUE second-order Dryden wind generator
    wind_gen = TrueDrydenWindGenerator(dt=dt, V=5.0, altitude=10.0, turbulence_level='moderate')
    drone = Quadcopter(dt)
    
    print("\nQUADCOPTER SPECS:")
    print(f"  Mass: {drone.mass:.3f} kg ({drone.mass*1000:.0f}g)")
    print(f"  Weight: {drone.weight:.2f}N")
    print(f"  Max thrust per motor: {drone.max_thrust:.1f}N")
    print(f"  Total thrust: {4*drone.max_thrust:.1f}N ({4*drone.max_thrust/drone.weight:.1f}x weight)")
    print(f"  Hover thrust per motor: {drone.hover_thrust:.2f}N")
    print(f"  Simulation: TRUE Second-Order Dryden Turbulence (MIL-F-8785C)")
    print("\nStarting simulation...")
    
    max_deviation = 0
    start_time = time.time()
    
    for i, t in enumerate(time_array):
        wind = wind_gen.update()
        state = drone.update(wind)
        
        pos = state[0:3]
        deviation = np.linalg.norm(pos)
        max_deviation = max(max_deviation, deviation)
        
        # Print status every 500 iterations
        if i % 500 == 0:
            pitch_deg = np.degrees(state[6])
            roll_deg = np.degrees(state[7])
            wind_kmh = wind * 3.6
            motor_pct = (np.mean(drone.motor_thrusts) / drone.max_thrust) * 100
            print(f"Time: {t:5.1f}s | Pos: [{pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f}] | "
                  f"Tilt: [{roll_deg:5.1f}, {pitch_deg:5.1f}] deg | "
                  f"Wind: [{wind_kmh[0]:5.1f}, {wind_kmh[1]:5.1f}, {wind_kmh[2]:5.1f}] km/h")
    
    print(f"\nSimulation complete in {time.time() - start_time:.1f} seconds!")
    print(f"Maximum deviation from origin: {max_deviation:.3f}m")
    
    plot_results(drone, time_array, wind_gen, dt)

def plot_results(drone, time_array, wind_gen, dt):
    positions = np.array(drone.positions)
    angles = np.degrees(np.array(drone.angles))
    motors = np.array(drone.motor_history)
    forces = np.array(drone.forces)
    
    wind_history = np.array(wind_gen.wind_history)
    wind_kmh = wind_history * 3.6
    
    # Create figure with 2x4 grid
    fig = plt.figure(figsize=(20, 12))
    
    # 1. POSITION PLOT
    ax1 = plt.subplot(2, 4, 1)
    ax1.plot(time_array, positions[:, 0], 'b-', linewidth=2, label='X Position')
    ax1.plot(time_array, positions[:, 1], 'g-', linewidth=2, label='Y Position')
    ax1.plot(time_array, positions[:, 2], 'r-', linewidth=2, label='Z Position')
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax1.set_ylabel('Position (m)')
    ax1.set_title('DRONE POSITION vs TIME')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_ylim([-0.3, 0.3])
    
    # 2. WIND COMPONENTS
    ax2 = plt.subplot(2, 4, 2)
    ax2.plot(time_array, wind_kmh[:, 0], 'b-', alpha=0.7, linewidth=1, label='Wind X (1st order)')
    ax2.plot(time_array, wind_kmh[:, 1], 'g-', alpha=0.7, linewidth=1, label='Wind Y (2nd order)')
    ax2.plot(time_array, wind_kmh[:, 2], 'r-', alpha=0.7, linewidth=1, label='Wind Z (2nd order)')
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.set_ylabel('Wind Speed (km/h)')
    ax2.set_title('TRUE DRYDEN TURBULENCE')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.set_ylim([-15, 15])
    
    # 3. TILT ANGLES
    ax3 = plt.subplot(2, 4, 3)
    ax3.plot(time_array, angles[:, 1], 'b-', linewidth=2, label='Pitch')
    ax3.plot(time_array, angles[:, 0], 'g-', linewidth=2, label='Roll')
    ax3.axhline(y=0, color='k', alpha=0.3)
    ax3.axhline(y=45, color='r', linestyle='--', alpha=0.5, label='Max ±45°')
    ax3.axhline(y=-45, color='r', linestyle='--', alpha=0.5)
    ax3.set_ylabel('Tilt Angle (deg)')
    ax3.set_title('PITCH AND ROLL ANGLES')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right', fontsize=8)
    ax3.set_ylim([-50, 50])
    
    # 4. FORCES
    ax4 = plt.subplot(2, 4, 4)
    ax4.plot(time_array, forces[:, 0], 'b-', alpha=0.7, linewidth=1, label='Force X')
    ax4.plot(time_array, forces[:, 1], 'g-', alpha=0.7, linewidth=1, label='Force Y')
    ax4.plot(time_array, forces[:, 2], 'r-', alpha=0.7, linewidth=1, label='Force Z')
    ax4.axhline(y=0, color='k', alpha=0.3)
    ax4.set_ylabel('Force (N)')
    ax4.set_title('FORCES ON DRONE')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper right', fontsize=7)
    
    # 5. DISTANCE FROM ORIGIN
    distance = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2 + positions[:, 2]**2)
    ax5 = plt.subplot(2, 4, 5)
    ax5.plot(time_array, distance, 'purple', linewidth=3)
    ax5.axhline(y=0, color='k', alpha=0.3)
    ax5.axhline(y=0.1, color='g', linestyle='--', linewidth=2, label='0.1m tolerance')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Distance from Origin (m)')
    ax5.set_title('TOTAL ERROR vs TIME')
    ax5.grid(True, alpha=0.3)
    ax5.legend(loc='upper right', fontsize=8)
    ax5.set_ylim([0, 0.5])
    
    # 6. POWER SPECTRAL DENSITY
    ax6 = plt.subplot(2, 4, 6)
    
    # Calculate empirical PSD from simulation
    fs = 1/dt
    
    f_u, P_u = welch(wind_history[:, 0], fs=fs, nperseg=min(1024, len(wind_history)), scaling='density')
    f_v, P_v = welch(wind_history[:, 1], fs=fs, nperseg=min(1024, len(wind_history)), scaling='density')
    f_w, P_w = welch(wind_history[:, 2], fs=fs, nperseg=min(1024, len(wind_history)), scaling='density')
    
    # Calculate theoretical PSD
    f_theory = np.logspace(-2, np.log10(fs/2), 200)
    Phi_u_theory, Phi_v_theory, Phi_w_theory = wind_gen.theoretical_psd(f_theory)
    
    # Plot empirical PSD
    ax6.loglog(f_u[f_u>0], P_u[f_u>0], 'b-', alpha=0.5, linewidth=1, label='X (Empirical)')
    ax6.loglog(f_v[f_v>0], P_v[f_v>0], 'g-', alpha=0.5, linewidth=1, label='Y (Empirical)')
    ax6.loglog(f_w[f_w>0], P_w[f_w>0], 'r-', alpha=0.5, linewidth=1, label='Z (Empirical)')
    
    # Plot theoretical PSD
    ax6.loglog(f_theory, Phi_u_theory, 'b--', linewidth=2, alpha=0.8, label='X (Theory, 1st order)')
    ax6.loglog(f_theory, Phi_v_theory, 'g--', linewidth=2, alpha=0.8, label='Y (Theory, 2nd order)')
    ax6.loglog(f_theory, Phi_w_theory, 'r--', linewidth=2, alpha=0.8, label='Z (Theory, 2nd order)')
    
    ax6.set_xlabel('Frequency (Hz)')
    ax6.set_ylabel('Power Spectral Density (m²/s³)')
    ax6.set_title('TRUE DRYDEN PSD: EMPIRICAL vs THEORETICAL')
    ax6.grid(True, alpha=0.3, which='both')
    ax6.legend(loc='upper right', fontsize=7)
    ax6.set_xlim([0.01, fs/2])
    
    # 7. MOTOR THRUSTS
    ax7 = plt.subplot(2, 4, 7)
    motor_pct = (motors / drone.max_thrust) * 100
    ax7.plot(time_array, motor_pct[:, 0], 'b-', alpha=0.7, linewidth=1, label='Motor 1')
    ax7.plot(time_array, motor_pct[:, 1], 'g-', alpha=0.7, linewidth=1, label='Motor 2')
    ax7.plot(time_array, motor_pct[:, 2], 'r-', alpha=0.7, linewidth=1, label='Motor 3')
    ax7.plot(time_array, motor_pct[:, 3], 'orange', alpha=0.7, linewidth=1, label='Motor 4')
    hover_pct = (drone.hover_thrust / drone.max_thrust) * 100
    ax7.axhline(y=hover_pct, color='k', linestyle='--', linewidth=2, 
                label=f'Hover ({hover_pct:.0f}%)')
    ax7.axhline(y=100, color='r', linestyle='--', alpha=0.5, linewidth=1, label='100% Limit')
    ax7.set_xlabel('Time (s)')
    ax7.set_ylabel('Motor Thrust (%)')
    ax7.set_title('MOTOR THRUSTS vs TIME')
    ax7.grid(True, alpha=0.3)
    ax7.legend(loc='upper right', fontsize=6)
    ax7.set_ylim([0, 110])
    
    # 8. 3D TRAJECTORY
    ax8 = plt.subplot(2, 4, 8, projection='3d')
    ax8.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
             'b-', alpha=0.7, linewidth=1)
    ax8.scatter(0, 0, 0, c='red', s=100, marker='*', label='Target (0,0,0)')
    ax8.set_xlabel('X (m)')
    ax8.set_ylabel('Y (m)')
    ax8.set_zlabel('Z (m)')
    ax8.set_title('3D TRAJECTORY')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    ax8.set_xlim([-0.5, 0.5])
    ax8.set_ylim([-0.5, 0.5])
    ax8.set_zlim([-0.3, 0.3])
    
    plt.suptitle(f'500g QUADCOPTER WITH TRUE SECOND-ORDER DRYDEN TURBULENCE (MIL-F-8785C)', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
    
    # ========== TURBULENCE STATISTICS ==========
    print("\n" + "=" * 80)
    print("TRUE DRYDEN TURBULENCE STATISTICS")
    print("=" * 80)
    
    print(f"\nTurbulence Statistics (m/s):")
    print(f"  X-direction (1st order):")
    print(f"    Mean = {np.mean(wind_history[:, 0]):.4f}, Std = {np.std(wind_history[:, 0]):.4f}")
    print(f"    Theoretical σ = {wind_gen.sigma_u:.4f}, Ratio = {np.std(wind_history[:, 0])/wind_gen.sigma_u:.3f}")
    
    print(f"\n  Y-direction (2nd order):")
    print(f"    Mean = {np.mean(wind_history[:, 1]):.4f}, Std = {np.std(wind_history[:, 1]):.4f}")
    print(f"    Theoretical σ = {wind_gen.sigma_v:.4f}, Ratio = {np.std(wind_history[:, 1])/wind_gen.sigma_v:.3f}")
    
    print(f"\n  Z-direction (2nd order):")
    print(f"    Mean = {np.mean(wind_history[:, 2]):.4f}, Std = {np.std(wind_history[:, 2]):.4f}")
    print(f"    Theoretical σ = {wind_gen.sigma_w:.4f}, Ratio = {np.std(wind_history[:, 2])/wind_gen.sigma_w:.3f}")
    
    # ========== PERFORMANCE ANALYSIS ==========
    print("\n" + "=" * 80)
    print("PERFORMANCE ANALYSIS")
    print("=" * 80)
    
    final_pos = positions[-1]
    final_dist = np.linalg.norm(final_pos)
    avg_dist = np.mean(distance)
    max_dist = np.max(distance)
    
    print(f"\nPosition Statistics:")
    print(f"  Average Distance: {avg_dist:.4f} m")
    print(f"  Maximum Distance: {max_dist:.4f} m")
    print(f"  Final Position: [{final_pos[0]:.4f}, {final_pos[1]:.4f}, {final_pos[2]:.4f}] m")
    
    success_x = abs(final_pos[0]) < 0.1
    success_y = abs(final_pos[1]) < 0.1
    success_z = abs(final_pos[2]) < 0.1
    
    print(f"\nAxis Performance (0.1m tolerance):")
    print(f"  X-axis: {'✓ PASS' if success_x else '✗ FAIL'} ({final_pos[0]:.3f}m)")
    print(f"  Y-axis: {'✓ PASS' if success_y else '✗ FAIL'} ({final_pos[1]:.3f}m)")
    print(f"  Z-axis: {'✓ PASS' if success_z else '✗ FAIL'} ({final_pos[2]:.3f}m)")
    
    if success_x and success_y and success_z:
        print("\n✅ SUCCESS! Drone maintains position despite true Dryden turbulence!")
    else:
        print("\n⚠️  Controller tuning needed for some axes")
    
    # Motor usage statistics
    avg_motor_pct = np.mean(motor_pct)
    max_motor_pct = np.max(motor_pct)
    print(f"\nMotor Usage:")
    print(f"  Average thrust: {avg_motor_pct:.1f}%")
    print(f"  Maximum thrust: {max_motor_pct:.1f}%")
    print(f"  Hover level: {hover_pct:.1f}%")
    print(f"  Thrust margin: {100 - max_motor_pct:.1f}% remaining")
    
    print(f"\nTurbulence Summary:")
    print(f"  Model: TRUE Second-Order Dryden (MIL-F-8785C)")
    print(f"  Level: {wind_gen.turbulence_level}")
    print(f"  Airspeed V: {wind_gen.V:.1f} m/s")
    print(f"  Scale lengths: L_u={wind_gen.L_u:.1f}m, L_v={wind_gen.L_v:.1f}m, L_w={wind_gen.L_w:.1f}m")

if __name__ == "__main__":
    main()