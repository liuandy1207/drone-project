import numpy as np
import matplotlib.pyplot as plt
import time

print("=" * 80)
print("QUADCOPTER SIMULATION: LINEARIZED NAVIER-STOKES")
print("=" * 80)

# ========== LNS WIND GENERATOR ==========
class LNSWindGenerator:
    def __init__(self, dt=0.01, mean_flow=None):
        self.dt = dt

        #Initialize deviation from mean flow, u', to 0
        self.wind = np.zeros(3)  
        self.wind_history = []

        # Matrix A representing velocity decay and wind shear, assumes slower decay in z-direction
        self.A = np.array([
            [-0.2, 0.05, 0.0],   
            [0.05, -0.2, 0.0],   
            [0.0, 0.0, -0.1]      
        ])

        # Matrix B representing stochastic forcing
        self.B = np.array([
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.2]
        ])

        # Set maximum deviations in x and y directions
        self.max_horizontal = 2.78  
        self.max_vertical = 0.56 

        # Set mean flow to 0
        self.mean_flow = np.zeros(3)

    def update(self):
        # Randomly generated 3-D Wiener increments for stochastic forcing
        dW = np.random.randn(3) * np.sqrt(self.dt)

        # Calculate position update using the Linearized Navier-Stokes equation
        self.wind += self.A @ self.wind * self.dt + self.B @ dW

        # Apply maximum deviation limits
        self.wind[0:2] = np.clip(self.wind[0:2], -self.max_horizontal, self.max_horizontal)
        self.wind[2]   = np.clip(self.wind[2],   -self.max_vertical,   self.max_vertical)

        # Save history
        self.wind_history.append(self.wind.copy())

        # Return total wind, which is the sum of mean flow and fluctuations
        return self.mean_flow + self.wind

# ========== QUADCOPTER ==========
class Quadcopter:
    def __init__(self, dt=0.01):
        # Physical properties
        self.mass = 0.51                     # mass of the quadcopter (kg)
        self.weight = 5.0                    # gravitational force mg (N)

        self.dt = dt                         # simulation timestep (s)

        # State vector: [x, y, z, vx, vy, vz, pitch, roll]
        self.state = np.zeros(8)

        # Motor thrust limits
        self.max_thrust = 2.5           # maximum thrust a motor can output
        self.hover_thrust = self.weight / 4  # thrust needed per motor to hover
        self.motor_thrusts = np.array([self.hover_thrust] * 4)

        # Position controller gains (PD + integral on z)
        self.kp_pos = np.array([8.0, 8.0, 20.0])   # proportional gains (x,y,z)
        self.kd_pos = np.array([4.0, 4.0, 10.0])   # derivative gains (x,y,z)
        self.ki_z = 0.5                            # integral gain for altitude
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
        self.integral_z = np.clip(self.integral_z, -1, 1)
        desired_acc[2] += self.ki_z * self.integral_z

        # Convert acceleration into required forces
        total_force_needed = self.mass * desired_acc

        # Force of gravity
        total_force_needed[2] += self.weight

        # Add wind disturbance compensation
        total_force_needed += -wind * 0.3

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
        angle_tau = 0.08
        self.state[6] += (desired_pitch - pitch) * self.dt / angle_tau
        self.state[7] += (desired_roll - roll) * self.dt / angle_tau

        # Update angles after dynamics
        pitch = self.state[6]
        roll = self.state[7]

        # Compute thrust for 4 motors
        base_thrust = thrust_mag_needed / 4
        pitch_adj = pitch 
        roll_adj = roll 

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
        drag_coeff = 0.2
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

# ========== MAIN SIMULATION ==========#
def main():
    dt = 0.01
    duration = 30
    time_array = np.arange(0, duration, dt)

    # Mean wind assumed tos be zero 
    mean_flow = np.array([0.0, 0.0, 0.0])
    wind_gen = LNSWindGenerator(dt, mean_flow=mean_flow)
    drone = Quadcopter(dt)

    max_deviation = 0
    start_time = time.time()

    for i, t in enumerate(time_array):
        #Generate LNS wind
        wind = wind_gen.update()          
        
        #Update drone position accordingly
        state = drone.update(wind)       

        pos = state[0:3]
        deviation = np.linalg.norm(pos)
        max_deviation = max(max_deviation, deviation)

        if i % 500 == 0:
            pitch_deg = np.degrees(state[6])
            roll_deg = np.degrees(state[7])
            print(f"Time: {t:5.1f}s | Pos: [{pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f}] | "
                  f"Tilt: [{roll_deg:5.1f}, {pitch_deg:5.1f}] deg | Dist: {deviation:5.3f}m")

    print(f"\nSimulation complete in {time.time() - start_time:.1f} seconds!")
    print(f"Maximum deviation from origin: {max_deviation:.3f}m")

    plot_results(drone, time_array, wind_gen)

# ========== PLOTTING FUNCTION ==========
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
    ax2.set_title('WIND COMPONENTS')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=8)
    
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
    ax7.plot(time_array, motors[:, 0], 'b-', alpha=0.7, linewidth=1, label='Motor 1')
    ax7.plot(time_array, motors[:, 1], 'g-', alpha=0.7, linewidth=1, label='Motor 2')
    ax7.plot(time_array, motors[:, 2], 'r-', alpha=0.7, linewidth=1, label='Motor 3')
    ax7.plot(time_array, motors[:, 3], 'orange', alpha=0.7, linewidth=1, label='Motor 4')
    ax7.axhline(y=drone.hover_thrust, color='k', linestyle='--', linewidth=2, 
                label=f'Hover ({drone.hover_thrust:.2f}N)')
    ax7.set_xlabel('Time (s)')
    ax7.set_ylabel('Motor Thrust (N)')
    ax7.set_title('MOTOR THRUSTS vs TIME')
    ax7.grid(True, alpha=0.3)
    ax7.legend(loc='upper right', fontsize=6)
    
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
    
    plt.suptitle('QUADCOPTER SIMULATION WITH COMPLETE WIND ANALYSIS', 
        fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
