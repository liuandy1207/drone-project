import numpy as np
import matplotlib.pyplot as plt
import time

print("=" * 80)
print("QUADCOPTER SIMULATION - SECOND ORDER DRYDEN TURBULENCE")
print("=" * 80)

# ========== SECOND ORDER DRYDEN WIND GENERATOR ==========
class SecondOrderDrydenWindGenerator:
    def __init__(self, dt=0.01):
        self.dt = dt
        self.wind = np.zeros(3)
        self.wind_history = []
        
        # Dryden model parameters - SAME AS ORIGINAL
        # Horizontal: sigma = 1.2 m/s (~4.3 km/h) -> most values within ±8-10 km/h
        # Vertical: sigma = 0.1 m/s (~0.36 km/h) -> most values within ±0.7 km/h (SMALL!)
        self.sigma_u = 19.2    # Horizontal X
        self.sigma_v = 19.2    # Horizontal Y  
        self.sigma_w = 3.1    # Vertical Z - VERY SMALL
        
        # Time constants - SAME AS ORIGINAL
        self.tau_u = 2.0  # X wind time constant
        self.tau_v = 2.0  # Y wind time constant
        self.tau_w = 4.0  # Z wind time constant - longer = slower, more stable
        
        # Second order filter states: [position, velocity]
        self.u_state = np.zeros(2)  # [u, du/dt]
        self.v_state = np.zeros(2)  # [v, dv/dt]
        self.w_state = np.zeros(2)  # [w, dw/dt]
        
        # Damping ratios for second order filters
        # 0.7 gives critical damping (smooth, no oscillations)
        self.zeta = 0.7
        
        # Pre-compute filter coefficients for second order system
        # For a second order system: d²x/dt² + 2ζω dx/dt + ω²x = K * noise
        # where ω = 1/τ (natural frequency)
        
        # Natural frequencies
        omega_u = 1.0 / self.tau_u
        omega_v = 1.0 / self.tau_v
        omega_w = 1.0 / self.tau_w
        
        # Gains to achieve desired standard deviation
        # For second order system: σ = K / sqrt(4ζω³)
        self.K_u = self.sigma_u * np.sqrt(4 * self.zeta * omega_u**3)
        self.K_v = self.sigma_v * np.sqrt(4 * self.zeta * omega_v**3)
        self.K_w = self.sigma_w * np.sqrt(4 * self.zeta * omega_w**3)
        
        # Discrete-time update matrices (using Euler method)
        # For small dt: x[k+1] ≈ (I + A*dt) * x[k] + B*dt * noise[k]
        # where A = [[0, 1], [-ω², -2ζω]] and B = [0, K]
        
        # X component
        self.A_u = np.array([[0, 1],
                             [-omega_u**2, -2*self.zeta*omega_u]])
        self.B_u = np.array([0, self.K_u])
        
        # Y component
        self.A_v = np.array([[0, 1],
                             [-omega_v**2, -2*self.zeta*omega_v]])
        self.B_v = np.array([0, self.K_v])
        
        # Z component
        self.A_w = np.array([[0, 1],
                             [-omega_w**2, -2*self.zeta*omega_w]])
        self.B_w = np.array([0, self.K_w])
        
        print(f"\nSECOND ORDER DRYDEN MODEL (same statistics):")
        print(f"  Horizontal: sigma={self.sigma_u:.2f} m/s (±{self.sigma_u*3.6:.1f} km/h)")
        print(f"  Vertical: sigma={self.sigma_w:.2f} m/s (±{self.sigma_w*3.6:.1f} km/h)")
        print(f"  Time constants: tau_u={self.tau_u:.1f}s, tau_v={self.tau_v:.1f}s, tau_w={self.tau_w:.1f}s")
        print(f"  Filter order: 2nd order (previously 1st order)")
        
    def update(self):
        # Generate white Gaussian noise - SAME AS ORIGINAL
        noise_u = np.random.randn()
        noise_v = np.random.randn()
        noise_w = np.random.randn()
        
        # Update X component (second order)
        # x[k+1] = (I + A*dt) * x[k] + B*dt * noise[k]
        I_plus_A_dt = np.eye(2) + self.A_u * self.dt
        self.u_state = np.dot(I_plus_A_dt, self.u_state) + self.B_u * noise_u * self.dt
        u_wind = self.u_state[0]  # First element is wind speed
        
        # Update Y component (second order)
        I_plus_A_dt = np.eye(2) + self.A_v * self.dt
        self.v_state = np.dot(I_plus_A_dt, self.v_state) + self.B_v * noise_v * self.dt
        v_wind = self.v_state[0]
        
        # Update Z component (second order)
        I_plus_A_dt = np.eye(2) + self.A_w * self.dt
        self.w_state = np.dot(I_plus_A_dt, self.w_state) + self.B_w * noise_w * self.dt
        w_wind = self.w_state[0]
        
        # Set wind components - SAME AS ORIGINAL
        self.wind[0] = u_wind
        self.wind[1] = v_wind
        self.wind[2] = w_wind * 0.5  # Additional scaling to ensure it's small
        
        # Add occasional horizontal gusts (keep vertical calm) - SAME AS ORIGINAL
        if np.random.rand() < 0.003:  # 0.3% chance each timestep
            gust_magnitude = np.random.uniform(1.0, 2.5)
            gust_direction = np.random.uniform(0, 2*np.pi)
            self.wind[0] += gust_magnitude * np.cos(gust_direction)
            self.wind[1] += gust_magnitude * np.sin(gust_direction)
            # No vertical component to gusts
        
        self.wind_history.append(self.wind.copy())
        return self.wind

# ========== QUADCOPTER (EXACTLY THE SAME) ==========
class Quadcopter:
    def __init__(self, dt=0.01):
        # Physical properties - CHANGED TO 0.5 kg (500g)
        self.mass = 0.5                    # mass of the quadcopter (kg) - 500g
        self.weight = self.mass * 9.81    # gravitational force mg (N) - using 9.81 m/s²

        self.dt = dt                         # simulation timestep (s)

        # State vector: [x, y, z, vx, vy, vz, pitch, roll]
        self.state = np.zeros(8)

        # Motor thrust limits - increased for heavier drone
        self.max_thrust = 3.0           # maximum thrust a motor can output
        self.hover_thrust = self.weight / 4  # thrust needed per motor to hover
        self.motor_thrusts = np.array([self.hover_thrust] * 4)

        # Position controller gains (PD + integral on z) - adjusted for heavier drone
        self.kp_pos = np.array([12.0, 12.0, 30.0])   # proportional gains (x,y,z) - increased
        self.kd_pos = np.array([6.0, 6.0, 15.0])     # derivative gains (x,y,z) - increased
        self.ki_z = 0.8                            # integral gain for altitude - increased
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
        total_force_needed += -wind * 0.5  # Increased compensation for heavier drone

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
        angle_tau = 0.1  # Slightly slower for heavier drone
        self.state[6] += (desired_pitch - pitch) * self.dt / angle_tau
        self.state[7] += (desired_roll - roll) * self.dt / angle_tau

        # Update angles after dynamics
        pitch = self.state[6]
        roll = self.state[7]

        # Compute thrust for 4 motors
        base_thrust = thrust_mag_needed / 4
        pitch_adj = pitch * 0.8  # Reduced adjustment for stability
        roll_adj = roll * 0.8    # Reduced adjustment for stability

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

        # Parameters for drag force calculated with linear approximation
        drag_coeff = 0.25  # Increased for heavier drone
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

        # Log changes in position and forces
        self.positions.append(pos.copy())
        self.angles.append([pitch, roll])
        self.motor_history.append(self.motor_thrusts.copy())
        self.forces.append(total_force.copy())

        return self.state
    
# ========== MAIN SIMULATION (ONLY WIND GENERATOR CHANGED) ==========
def main():
    dt = 0.01
    duration = 30 #Total simulation time in seconds
    time_array = np.arange(0, duration, dt)
    
    # ONLY THIS LINE CHANGED: Using SecondOrderDrydenWindGenerator instead of DrydenWindGenerator
    wind_gen = SecondOrderDrydenWindGenerator(dt)
    drone = Quadcopter(dt)
    
    print("\nQUADCOPTER SPECS:")
    print(f"  Mass: {drone.mass:.3f} kg ({drone.mass*1000:.0f}g)")
    print(f"  Weight: {drone.weight:.2f}N")
    print(f"  Max thrust per motor: {drone.max_thrust:.1f}N")
    print(f"  Total thrust: {4*drone.max_thrust:.1f}N ({4*drone.max_thrust/drone.weight:.1f}x weight)")
    print(f"  Hover thrust per motor: {drone.hover_thrust:.2f}N")
    print(f"\nWIND CONDITIONS (Second Order Dryden Model):")
    print(f"  Horizontal: ~{wind_gen.sigma_u*3.6*2:.1f} km/h peak-to-peak (±{wind_gen.sigma_u*3.6:.1f} km/h)")
    print(f"  Vertical: ~{wind_gen.sigma_w*3.6*2:.1f} km/h peak-to-peak (±{wind_gen.sigma_w*3.6:.1f} km/h)")
    print("\nStarting simulation...")
    
    max_deviation = 0
    start_time = time.time()
    

    for i, t in enumerate(time_array):
        wind = wind_gen.update() #Generate wind vector
        state = drone.update(wind) #Update drone forces with new wind vector
        
        pos = state[0:3] #Drone position
        deviation = np.linalg.norm(pos) #Drone distance from origin
        max_deviation = max(max_deviation, deviation)
        
        #Print drone position every 500 iterations
        if i % 500 == 0:
            pitch_deg = np.degrees(state[6])
            roll_deg = np.degrees(state[7])
            wind_kmh = wind * 3.6
            motor_pct = (np.mean(drone.motor_thrusts) / drone.max_thrust) * 100
            print(f"Time: {t:5.1f}s | Pos: [{pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f}] | "
                  f"Tilt: [{roll_deg:5.1f}, {pitch_deg:5.1f}] deg | "
                  f"Wind: [{wind_kmh[0]:5.1f}, {wind_kmh[1]:5.1f}, {wind_kmh[2]:5.1f}] km/h | "
                  f"Motors: {motor_pct:4.0f}%")
    
    print(f"\nSimulation complete in {time.time() - start_time:.1f} seconds!")
    print(f"Maximum deviation from origin: {max_deviation:.3f}m")
    
    plot_results(drone, time_array, wind_gen)

def plot_results(drone, time_array, wind_gen):
    positions = np.array(drone.positions)
    angles = np.degrees(np.array(drone.angles))
    motors = np.array(drone.motor_history)
    forces = np.array(drone.forces)
    
    wind_history = np.array(wind_gen.wind_history)
    wind_kmh = wind_history * 3.6
    
    # Create figure with 2x4 grid
    fig = plt.figure(figsize=(20, 10))
    
    # 1. POSITION PLOT (Top Left)
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
    
    # 2. WIND COMPONENTS (Top Center-Left)
    ax2 = plt.subplot(2, 4, 2)
    ax2.plot(time_array, wind_kmh[:, 0], 'b-', alpha=0.7, linewidth=1, label='Wind X')
    ax2.plot(time_array, wind_kmh[:, 1], 'g-', alpha=0.7, linewidth=1, label='Wind Y')
    ax2.plot(time_array, wind_kmh[:, 2], 'r-', alpha=0.7, linewidth=1, label='Wind Z')
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.set_ylabel('Wind Speed (km/h)')
    ax2.set_title('WIND COMPONENTS (2nd Order Dryden Model)')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=8)
    # Set y-limits to show that vertical wind is much smaller
    ax2.set_ylim([-15, 15])
    
    # 3. TILT ANGLES (Top Center-Right)
    ax3 = plt.subplot(2, 4, 3)
    ax3.plot(time_array, angles[:, 1], 'b-', linewidth=2, label='Pitch')
    ax3.plot(time_array, angles[:, 0], 'g-', linewidth=2, label='Roll')
    ax3.axhline(y=0, color='k', alpha=0.3)
    ax3.axhline(y=45, color='r', linestyle='--', alpha=0.5, label='Max +/-45')
    ax3.axhline(y=-45, color='r', linestyle='--', alpha=0.5)
    ax3.set_ylabel('Tilt Angle (deg)')
    ax3.set_title('PITCH AND ROLL ANGLES')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right', fontsize=8)
    ax3.set_ylim([-50, 50])
    
    # 4. FORCES (Top Right)
    ax4 = plt.subplot(2, 4, 4)
    ax4.plot(time_array, forces[:, 0], 'b-', alpha=0.7, linewidth=1, label='Force X')
    ax4.plot(time_array, forces[:, 1], 'g-', alpha=0.7, linewidth=1, label='Force Y')
    ax4.plot(time_array, forces[:, 2], 'r-', alpha=0.7, linewidth=1, label='Force Z')
    ax4.axhline(y=0, color='k', alpha=0.3)
    ax4.set_ylabel('Force (N)')
    ax4.set_title('FORCES ON DRONE')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper right', fontsize=7)
    
    # 5. DISTANCE FROM ORIGIN (Bottom Left)
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
    
    # 6. WIND DIRECTION (Bottom Center-Left)
    ax6 = plt.subplot(2, 4, 6)
    wind_direction = np.degrees(np.arctan2(wind_kmh[:, 1], wind_kmh[:, 0]))
    ax6.plot(time_array, wind_direction, 'purple', linewidth=1.5, alpha=0.7)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Wind Direction (deg)')
    ax6.set_title('WIND DIRECTION (0 deg = East)')
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim([-180, 180])
    
    # 7. MOTOR THRUSTS (Bottom Center-Right)
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
    
    # 8. 3D TRAJECTORY (Bottom Right)
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
    
    plt.suptitle(f'500g QUADCOPTER WITH SECOND ORDER DRYDEN TURBULENCE (SAME STATISTICS)', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
    
    # ========== WIND STATISTICS ==========
    print("\n" + "=" * 80)
    print("SECOND ORDER DRYDEN TURBULENCE STATISTICS")
    print("=" * 80)
    
    print(f"\nWind Speeds (km/h):")
    print(f"  X-direction: Mean = {np.mean(wind_kmh[:, 0]):.2f}, Std = {np.std(wind_kmh[:, 0]):.2f}, "
          f"Target σ = {wind_gen.sigma_u*3.6:.2f}")
    print(f"    Range: [{np.min(wind_kmh[:, 0]):.2f}, {np.max(wind_kmh[:, 0]):.2f}]")
    print(f"  Y-direction: Mean = {np.mean(wind_kmh[:, 1]):.2f}, Std = {np.std(wind_kmh[:, 1]):.2f}, "
          f"Target σ = {wind_gen.sigma_v*3.6:.2f}")
    print(f"    Range: [{np.min(wind_kmh[:, 1]):.2f}, {np.max(wind_kmh[:, 1]):.2f}]")
    print(f"  Z-direction: Mean = {np.mean(wind_kmh[:, 2]):.3f}, Std = {np.std(wind_kmh[:, 2]):.3f}, "
          f"Target σ = {wind_gen.sigma_w*3.6:.3f}")
    print(f"    Range: [{np.min(wind_kmh[:, 2]):.3f}, {np.max(wind_kmh[:, 2]):.3f}]")
    
    # Compare achieved vs target standard deviations
    print(f"\nStandard Deviation Match (should be close to 1.0):")
    print(f"  X: Achieved {np.std(wind_history[:, 0]):.3f} m/s / Target {wind_gen.sigma_u:.3f} m/s = {np.std(wind_history[:, 0])/wind_gen.sigma_u:.3f}")
    print(f"  Y: Achieved {np.std(wind_history[:, 1]):.3f} m/s / Target {wind_gen.sigma_v:.3f} m/s = {np.std(wind_history[:, 1])/wind_gen.sigma_v:.3f}")
    print(f"  Z: Achieved {np.std(wind_history[:, 2]):.3f} m/s / Target {wind_gen.sigma_w:.3f} m/s = {np.std(wind_history[:, 2])/wind_gen.sigma_w:.3f}")
    
    # Compare horizontal vs vertical
    print(f"\nWind Magnitude Comparison:")
    horizontal_wind = np.sqrt(wind_kmh[:, 0]**2 + wind_kmh[:, 1]**2)
    vertical_wind = np.abs(wind_kmh[:, 2])
    print(f"  Horizontal: Mean = {np.mean(horizontal_wind):.2f} km/h, Max = {np.max(horizontal_wind):.2f} km/h")
    print(f"  Vertical: Mean = {np.mean(vertical_wind):.3f} km/h, Max = {np.max(vertical_wind):.3f} km/h")
    print(f"  Ratio (Horizontal/Vertical): {np.mean(horizontal_wind)/np.mean(vertical_wind):.1f}:1")
    
    # ========== PERFORMANCE ANALYSIS ==========
    print("\n" + "=" * 80)
    print("PERFORMANCE ANALYSIS")
    print("=" * 80)
    
    final_pos = positions[-1]
    final_dist = np.linalg.norm(final_pos)
    avg_dist = np.mean(distance)
    max_dist = np.max(distance)
    
    print(f"Average Distance: {avg_dist:.4f} m")
    print(f"Maximum Distance: {max_dist:.4f} m")
    print(f"Final Position: [{final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f}] m")
    
    # Motor usage statistics
    avg_motor_pct = np.mean(motor_pct)
    max_motor_pct = np.max(motor_pct)
    print(f"\nMotor Usage:")
    print(f"  Average thrust: {avg_motor_pct:.1f}%")
    print(f"  Maximum thrust: {max_motor_pct:.1f}%")
    print(f"  Hover level: {hover_pct:.1f}%")
    
    # Thrust margin
    thrust_margin = 100 - max_motor_pct
    print(f"  Thrust margin: {thrust_margin:.1f}% remaining")
    
    print(f"\nMaximum Wind Encountered: {np.max(horizontal_wind):.1f} km/h horizontal, "
          f"{np.max(vertical_wind):.2f} km/h vertical")
    print(f"Drone Thrust/Weight Ratio: {4*drone.max_thrust/drone.weight:.1f}:1")
    
    # ========== SECOND ORDER VERIFICATION ==========
    print("\n" + "=" * 80)
    print("SECOND ORDER DYNAMICS VERIFICATION")
    print("=" * 80)
    
    # Check if vertical wind is indeed small
    vertical_stats = wind_kmh[:, 2]
    within_range = np.sum(np.abs(vertical_stats) < 1.0) / len(vertical_stats) * 100
    print(f"Vertical wind within ±1 km/h: {within_range:.1f}% of the time")
    print(f"Vertical wind within ±0.5 km/h: {np.sum(np.abs(vertical_stats) < 0.5) / len(vertical_stats) * 100:.1f}% of the time")
    
    print(f"\nKey difference from 1st order:")
    print(f"  • Wind changes more smoothly (less abrupt)")
    print(f"  • Same overall statistics (mean, std dev, range)")
    print(f"  • More realistic temporal correlation")
    print(f"  • Should produce similar drone performance")

if __name__ == "__main__":
    main()