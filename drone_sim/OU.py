################################################################################
#
#                           OU WIND SIMULATION
#
# Authors: Andy Liu, John Chen, Melyssa Choi, Yawen Zheng
################################################################################

import numpy as np
import matplotlib.pyplot as plt
import time

class OUWindGenerator:
    def __init__(self, dt=0.01):
        self.dt = dt              # time step for numerical methods (s)
        self.wind = np.zeros(3)   # creates a 3-component (xyz) wind vector 
        self.wind_history = []    # stores past wind vectors

    def update(self):
        # OU parameters
        theta = 0.15    # mean reversion rate (how strongly results are pulled to the average)
                        # (justification)
        sigma_h = 0.68  # horizontal noise intensity
        sigma_v = 0.13  # vertical noise intensity

        mu_h = 0        # mean horizontal wind speed
        mu_v = 0        # mean vertical wind speed
                        # set to zero because the simulation is centered at the origin

        # Euler–Maruyama Method: dW = θ(µ-W)dt + σ dB (Equation ?)
        # approximates a numerical solution of an SDE
        self.wind[0] += theta * (mu_h - self.wind[0]) * self.dt + \
                       sigma_h * np.sqrt(self.dt) * np.random.randn()
        self.wind[1] += theta * (mu_h - self.wind[1]) * self.dt + \
                       sigma_h * np.sqrt(self.dt) * np.random.randn()
        self.wind[2] += theta * (mu_v - self.wind[2]) * self.dt + \
                       sigma_v * np.sqrt(self.dt) * np.random.randn()
    
        self.wind_history.append(self.wind.copy())  # save generated wind vector to history
        return self.wind

class Quadcopter:
    def __init__(self, dt=0.01):
        self.dt = dt                    # time step (s)
        # physical properties
        self.mass = 0.25                # same as our real drone (kg)
        self.weight = self.mass * 9.81    # gravitaitonal force (N)

        # state vector: [x, y, z, vx, vy, vz, pitch, roll]
        self.state = np.zeros(8)

        # motor thrust specifications
        self.max_thrust = 2.5                   # N
        self.hover_thrust = self.weight / 4     # thrust per motor to hover
        self.motor_thrusts = np.array([self.hover_thrust] * 4)  # thrust in each motor

        # histories
        self.positions = []
        self.angles = []
        self.motor_history = []
        self.forces = []

        # proportial derivative (PD) control on position
        self.kp_pos = np.array([8.0, 8.0, 20.0])   # proportional gains (corrective pull towards origin)
        self.kd_pos = np.array([4.0, 4.0, 10.0])   # derivative gains (damping effects)
        self.integral_z = 0                        # accumulates z error for PID to avoid steady state error
        self.ki_z = 0.5                            # integral gain (corrective pull)

    def update(self, wind):
        # extract components
        pos = self.state[0:3]
        vel = self.state[3:6]
        pitch = self.state[6]
        roll = self.state[7]

        # PD controller calculations
        error = -pos    # error to eliminate to maintain position at origin (hover)
        desired_acc = self.kp_pos * error + self.kd_pos * (-vel)    # PD corrective acceleartion
        self.integral_z += error[2] * self.dt               # accumulating error for PID correction in z
        self.integral_z = np.clip(self.integral_z, -1, 1)   # clamp to avoid integrator windup
        desired_acc[2] += self.ki_z * self.integral_z       # adds the effect of the integral
        total_force_needed = self.mass * desired_acc
        total_force_needed[2] += self.weight    # compensate for gravity
        total_force_needed += -wind * 0.3       # consider wind effects
        thrust_mag_needed = np.linalg.norm(total_force_needed)

        if thrust_mag_needed > 0.01:            # avoid divide by zero
            thrust_dir = total_force_needed / thrust_mag_needed   # unit vector
            # limit tilt to be mathematically defined
            fx_ratio = np.clip(thrust_dir[0], -0.9, 0.9)
            fy_ratio = np.clip(thrust_dir[1], -0.9, 0.9)
            desired_pitch = np.arcsin(fx_ratio)
            desired_roll = -np.arcsin(fy_ratio)
        else: 
            desired_pitch = 0
            desired_roll = 0
            thrust_mag_needed = self.weight

        # limit tilt to be physically reasonable
        max_angle = np.radians(45)
        desired_pitch = np.clip(desired_pitch, -max_angle, max_angle)
        desired_roll = np.clip(desired_roll, -max_angle, max_angle)

        # update roll, pitch
        angle_tau = 0.08 # angle response time constant for gradual changes
        self.state[6] += (desired_pitch - pitch) * self.dt / angle_tau
        self.state[7] += (desired_roll - roll) * self.dt / angle_tau
        
        pitch = self.state[6]
        roll = self.state[7]
        base_thrust = thrust_mag_needed / 4
        pitch_adj = pitch 
        roll_adj = roll 

        # adjust thrust for the four motors
        self.motor_thrusts = np.array([
            base_thrust - roll_adj + pitch_adj,   # front-left
            base_thrust + roll_adj + pitch_adj,   # front-right
            base_thrust - roll_adj - pitch_adj,   # back-left
            base_thrust + roll_adj - pitch_adj    # back-right
        ])
        # prevent motor thrust from going to zero
        self.motor_thrusts = np.maximum(self.motor_thrusts, 0.1)

        # compute thrust forces
        total_thrust = np.sum(self.motor_thrusts)
        thrust_x = np.sin(pitch) * total_thrust
        thrust_y = -np.sin(roll) * total_thrust
        thrust_z = np.cos(pitch) * np.cos(roll) * total_thrust
        thrust_force = np.array([thrust_x, thrust_y, thrust_z])

        # linear drag force approximation
        drag_coeff = 0.2
        drag_force = -drag_coeff * (vel - wind)

        # gravity 
        gravity_force = np.array([0, 0, -self.weight])

        # net force and acceleration
        total_force = thrust_force + drag_force + gravity_force
        acceleration = total_force / self.mass

        # update velocity and position
        self.state[3:6] += acceleration * self.dt
        self.state[0:3] += self.state[3:6] * self.dt
        self.state[0:3] = np.clip(self.state[0:3], -10, 10) # bound on position for the simulation

        # save result to history
        self.positions.append(pos.copy())
        self.angles.append([pitch, roll])
        self.motor_history.append(self.motor_thrusts.copy())
        self.forces.append(total_force.copy())

        return self.state
    
def main():
    dt = 0.01
    duration = 30       # total simulation time
    time_array = np.arange(0, duration, dt)
    
    wind_gen = OUWindGenerator(dt)
    drone = Quadcopter(dt)
    
    print("\nQUADCOPTER SPECS:")
    print(f"  Weight: {drone.weight:.2f}N")
    print(f"  Max thrust: {4*drone.max_thrust:.0f}N ({4*drone.max_thrust/drone.weight:.0f}x weight)")
    print(f"  Wind: +/-10 km/h horizontal, +/-2-3 km/h vertical")
    print("\nStarting simulation...")
    
    max_deviation = 0
    start_time = time.time()
    
    for i, t in enumerate(time_array):  # for each time step
        wind = wind_gen.update()    # generate a wind vectors
        state = drone.update(wind)  # update drone state

        pos = state[0:3] # extrac position
        deviation = np.linalg.norm(pos) # extract distance from the origin
        max_deviation = max(max_deviation, deviation)
        
        # print progress every 500 time steps
        if i % 500 == 0:
            pitch_deg = np.degrees(state[6])
            roll_deg = np.degrees(state[7])
            print(f"Time: {t:5.1f}s | Pos: [{pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f}] | "
                  f"Tilt: [{roll_deg:5.1f}, {pitch_deg:5.1f}] deg | Dist: {deviation:5.3f}m")
    
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
    
    # ========== WIND STATISTICS ==========
    print("\n" + "=" * 80)
    print("WIND STATISTICS")
    print("=" * 80)
    
    print(f"\nWind Speeds (km/h):")
    print(f"  X-direction: Mean = {np.mean(wind_kmh[:, 0]):.1f}, Std = {np.std(wind_kmh[:, 0]):.1f}, Max = {np.max(np.abs(wind_kmh[:, 0])):.1f}")
    print(f"  Y-direction: Mean = {np.mean(wind_kmh[:, 1]):.1f}, Std = {np.std(wind_kmh[:, 1]):.1f}, Max = {np.max(np.abs(wind_kmh[:, 1])):.1f}")
    print(f"  Z-direction: Mean = {np.mean(wind_kmh[:, 2]):.1f}, Std = {np.std(wind_kmh[:, 2]):.1f}, Max = {np.max(np.abs(wind_kmh[:, 2])):.1f}")
    
    horizontal_wind = np.sqrt(wind_kmh[:, 0]**2 + wind_kmh[:, 1]**2)
    print(f"\nHorizontal Wind Magnitude:")
    print(f"  Mean: {np.mean(horizontal_wind):.1f} km/h")
    print(f"  Max: {np.max(horizontal_wind):.1f} km/h")
    
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
        
    print(f"\nMaximum Wind Encountered: {np.max(horizontal_wind):.1f} km/h")
    print(f"Drone Thrust/Weight Ratio: {4*drone.max_thrust/drone.weight:.0f}:1")

if __name__ == "__main__":
    main()