import numpy as np
import matplotlib.pyplot as plt
import time

print("=" * 80)
print("QUADCOPTER SIMULATION - COMPLETE WITH WIND PLOTS")
print("=" * 80)

# ========== OU WIND GENERATOR ==========
class OUWindGenerator:
    def __init__(self, dt=0.01):
        self.dt = dt
        self.wind = np.zeros(3)
        self.wind_history = []
        
    def update(self):
        theta = 0.15
        mu_h = 0
        sigma_h = 2.78
        mu_v = 0
        sigma_v = 0.56
        
        self.wind[0] += theta * (mu_h - self.wind[0]) * self.dt + \
                       sigma_h * np.sqrt(self.dt) * np.random.randn()
        self.wind[1] += theta * (mu_h - self.wind[1]) * self.dt + \
                       sigma_h * np.sqrt(self.dt) * np.random.randn()
        self.wind[2] += theta * (mu_v - self.wind[2]) * self.dt + \
                       sigma_v * np.sqrt(self.dt) * np.random.randn()
        
        self.wind[2] = np.clip(self.wind[2], -0.8, 0.8)
        
        self.wind_history.append(self.wind.copy())
        return self.wind

# ========== QUADCOPTER ==========
class Quadcopter:
    def __init__(self, dt=0.01):
        self.mass = 0.51
        self.weight = 5.0
        self.dt = dt
        
        self.state = np.zeros(8)
        
        self.max_thrust = 1000.0
        self.hover_thrust = self.weight / 4
        
        self.motor_thrusts = np.array([self.hover_thrust] * 4)
        
        self.kp_pos = np.array([8.0, 8.0, 20.0])
        self.kd_pos = np.array([4.0, 4.0, 10.0])
        
        self.ki_z = 0.5
        self.integral_z = 0
        
        self.positions = []
        self.angles = []
        self.motor_history = []
        self.forces = []
        
    def update(self, wind):
        pos = self.state[0:3]
        vel = self.state[3:6]
        pitch = self.state[6]
        roll = self.state[7]
        
        error = -pos
        
        desired_acc = self.kp_pos * error + self.kd_pos * (-vel)
        
        self.integral_z += error[2] * self.dt
        self.integral_z = np.clip(self.integral_z, -1, 1)
        desired_acc[2] += self.ki_z * self.integral_z
        
        max_acc = 15.0
        desired_acc = np.clip(desired_acc, -max_acc, max_acc)
        
        total_force_needed = self.mass * desired_acc
        total_force_needed[2] += self.weight
        total_force_needed += -wind * 0.3
        
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
        
        angle_tau = 0.08
        self.state[6] += (desired_pitch - pitch) * self.dt / angle_tau
        self.state[7] += (desired_roll - roll) * self.dt / angle_tau
        
        pitch = self.state[6]
        roll = self.state[7]
        
        base_thrust = thrust_mag_needed / 4
        pitch_adj = pitch * 5.0
        roll_adj = roll * 5.0
        
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
        drag_coeff = 0.2
        drag_force = -drag_coeff * (vel - wind)
        gravity_force = np.array([0, 0, -self.weight])
        
        total_force = thrust_force + drag_force + gravity_force
        acceleration = total_force / self.mass
        
        self.state[3:6] += acceleration * self.dt
        
        max_vel = 5.0
        vel_mag = np.linalg.norm(self.state[3:6])
        if vel_mag > max_vel:
            self.state[3:6] = self.state[3:6] / vel_mag * max_vel
        
        self.state[0:3] += self.state[3:6] * self.dt
        self.state[0:3] = np.clip(self.state[0:3], -10, 10)
        
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
    
    wind_gen = OUWindGenerator(dt)
    drone = Quadcopter(dt)
    
    print("\nQUADCOPTER SPECS:")
    print(f"  Weight: {drone.weight:.2f}N")
    print(f"  Max thrust: {4*drone.max_thrust:.0f}N ({4*drone.max_thrust/drone.weight:.0f}x weight)")
    print(f"  Wind: +/-10 km/h horizontal, +/-2-3 km/h vertical")
    print("\nStarting simulation...")
    
    max_deviation = 0
    start_time = time.time()
    
    for i, t in enumerate(time_array):
        wind = wind_gen.update()
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

def plot_results(drone, time_array, wind_gen):
    positions = np.array(drone.positions)
    angles = np.degrees(np.array(drone.angles))
    motors = np.array(drone.motor_history)
    forces = np.array(drone.forces)
    
    wind_history = np.array(wind_gen.wind_history)
    wind_kmh = wind_history * 3.6
    
    fig = plt.figure(figsize=(20, 18))
    
    # 1. WIND PLOTS
    ax1 = plt.subplot(4, 4, 1)
    ax1.plot(time_array, wind_kmh[:, 0], 'b-', linewidth=1.5, alpha=0.8)
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax1.axhline(y=10, color='r', linestyle='--', alpha=0.5)
    ax1.axhline(y=-10, color='r', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Wind Speed (km/h)')
    ax1.set_title('WIND X vs TIME')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([-15, 15])
    
    ax2 = plt.subplot(4, 4, 2)
    ax2.plot(time_array, wind_kmh[:, 1], 'g-', linewidth=1.5, alpha=0.8)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.axhline(y=10, color='r', linestyle='--', alpha=0.5)
    ax2.axhline(y=-10, color='r', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Wind Speed (km/h)')
    ax2.set_title('WIND Y vs TIME')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([-15, 15])
    
    ax3 = plt.subplot(4, 4, 3)
    ax3.plot(time_array, wind_kmh[:, 2], 'r-', linewidth=1.5, alpha=0.8)
    ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax3.axhline(y=3, color='orange', linestyle='--', alpha=0.5)
    ax3.axhline(y=-2, color='orange', linestyle='--', alpha=0.5)
    ax3.set_ylabel('Wind Speed (km/h)')
    ax3.set_title('VERTICAL WIND vs TIME')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([-5, 5])
    
    ax4 = plt.subplot(4, 4, 4)
    ax4.plot(time_array, wind_kmh[:, 0], 'b-', alpha=0.7, linewidth=1)
    ax4.plot(time_array, wind_kmh[:, 1], 'g-', alpha=0.7, linewidth=1)
    ax4.plot(time_array, wind_kmh[:, 2], 'r-', alpha=0.7, linewidth=1)
    ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax4.set_ylabel('Wind Speed (km/h)')
    ax4.set_title('ALL WIND COMPONENTS')
    ax4.grid(True, alpha=0.3)
    ax4.legend(['Wind X', 'Wind Y', 'Wind Z'], loc='upper right', fontsize=8)
    
    # 2. POSITION PLOTS
    ax5 = plt.subplot(4, 4, 5)
    ax5.plot(time_array, positions[:, 0], 'b-', linewidth=2)
    ax5.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax5.set_ylabel('X Position (m)')
    ax5.set_title('DRONE X POSITION vs TIME')
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim([-0.5, 0.5])
    
    ax6 = plt.subplot(4, 4, 6)
    ax6.plot(time_array, positions[:, 1], 'g-', linewidth=2)
    ax6.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax6.set_ylabel('Y Position (m)')
    ax6.set_title('DRONE Y POSITION vs TIME')
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim([-0.5, 0.5])
    
    ax7 = plt.subplot(4, 4, 7)
    ax7.plot(time_array, positions[:, 2], 'r-', linewidth=2)
    ax7.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax7.set_ylabel('Z Position (m)')
    ax7.set_title('DRONE ALTITUDE vs TIME')
    ax7.grid(True, alpha=0.3)
    ax7.set_ylim([-0.3, 0.3])
    
    # 3. MOTOR THRUST PERCENTAGE
    thrust_percent = (motors / drone.max_thrust) * 100
    hover_percent = (drone.hover_thrust / drone.max_thrust) * 100
    
    ax8 = plt.subplot(4, 4, 8)
    ax8.plot(time_array, thrust_percent[:, 0], 'b-', alpha=0.7, linewidth=1)
    ax8.plot(time_array, thrust_percent[:, 1], 'g-', alpha=0.7, linewidth=1)
    ax8.plot(time_array, thrust_percent[:, 2], 'r-', alpha=0.7, linewidth=1)
    ax8.plot(time_array, thrust_percent[:, 3], 'orange', alpha=0.7, linewidth=1)
    ax8.axhline(y=hover_percent, color='k', linestyle='--', linewidth=2, label=f'Hover ({hover_percent:.4f}%)')
    ax8.set_ylabel('Motor Thrust (%)')
    ax8.set_title('MOTOR THRUST PERCENTAGE vs TIME')
    ax8.grid(True, alpha=0.3)
    ax8.legend(loc='upper right', fontsize=7)
    ax8.set_ylim([hover_percent-0.01, hover_percent+0.02])
    
    # 4. TILT ANGLES
    ax9 = plt.subplot(4, 4, 9)
    ax9.plot(time_array, angles[:, 1], 'b-', linewidth=2, label='Pitch')
    ax9.plot(time_array, angles[:, 0], 'g-', linewidth=2, label='Roll')
    ax9.axhline(y=0, color='k', alpha=0.3)
    ax9.axhline(y=45, color='r', linestyle='--', alpha=0.5, label='Max +/-45')
    ax9.axhline(y=-45, color='r', linestyle='--', alpha=0.5)
    ax9.set_ylabel('Tilt Angle (deg)')
    ax9.set_title('PITCH AND ROLL ANGLES vs TIME')
    ax9.grid(True, alpha=0.3)
    ax9.legend(loc='upper right', fontsize=8)
    ax9.set_ylim([-50, 50])
    
    # 5. FORCES
    ax10 = plt.subplot(4, 4, 10)
    ax10.plot(time_array, forces[:, 0], 'b-', alpha=0.7, linewidth=1, label='Force X')
    ax10.plot(time_array, forces[:, 1], 'g-', alpha=0.7, linewidth=1, label='Force Y')
    ax10.plot(time_array, forces[:, 2], 'r-', alpha=0.7, linewidth=1, label='Force Z')
    ax10.axhline(y=0, color='k', alpha=0.3)
    ax10.axhline(y=drone.weight, color='purple', linestyle='--', alpha=0.5, label=f'Weight ({drone.weight}N)')
    ax10.set_ylabel('Force (N)')
    ax10.set_title('FORCES ON DRONE vs TIME')
    ax10.grid(True, alpha=0.3)
    ax10.legend(loc='upper right', fontsize=7)
    
    # 6. DISTANCE FROM ORIGIN
    distance = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2 + positions[:, 2]**2)
    ax11 = plt.subplot(4, 4, 11)
    ax11.plot(time_array, distance, 'purple', linewidth=3)
    ax11.axhline(y=0, color='k', alpha=0.3)
    ax11.axhline(y=0.1, color='g', linestyle='--', linewidth=2, label='0.1m tolerance')
    ax11.set_xlabel('Time (s)')
    ax11.set_ylabel('Distance from Origin (m)')
    ax11.set_title('TOTAL ERROR vs TIME')
    ax11.grid(True, alpha=0.3)
    ax11.legend(loc='upper right', fontsize=8)
    ax11.set_ylim([0, 0.5])
    
    # 7. CONTROL RELATIONSHIPS
    ax12 = plt.subplot(4, 4, 12)
    scatter = ax12.scatter(positions[:, 0], angles[:, 1], c=time_array, 
                          cmap='coolwarm', alpha=0.7, s=10)
    ax12.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax12.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax12.set_xlabel('X Position (m)')
    ax12.set_ylabel('Pitch Angle (deg)')
    ax12.set_title('CONTROL: X Position vs Pitch')
    ax12.grid(True, alpha=0.3)
    ax12.set_xlim([-0.5, 0.5])
    ax12.set_ylim([-50, 50])
    
    # 8. WIND MAGNITUDE
    ax13 = plt.subplot(4, 4, 13)
    wind_magnitude = np.sqrt(wind_kmh[:, 0]**2 + wind_kmh[:, 1]**2)
    ax13.plot(time_array, wind_magnitude, 'blue', linewidth=2, alpha=0.7, label='Horizontal Wind')
    ax13.plot(time_array, np.abs(wind_kmh[:, 2]), 'red', linewidth=2, alpha=0.7, label='Vertical Wind')
    ax13.axhline(y=10, color='b', linestyle='--', alpha=0.5, label='10 km/h typical')
    ax13.axhline(y=3, color='r', linestyle='--', alpha=0.5, label='3 km/h typical')
    ax13.set_xlabel('Time (s)')
    ax13.set_ylabel('Wind Speed (km/h)')
    ax13.set_title('WIND MAGNITUDE vs TIME')
    ax13.grid(True, alpha=0.3)
    ax13.legend(loc='upper right', fontsize=7)
    
    # 9. WIND DIRECTION
    ax14 = plt.subplot(4, 4, 14)
    wind_direction = np.degrees(np.arctan2(wind_kmh[:, 1], wind_kmh[:, 0]))
    ax14.plot(time_array, wind_direction, 'purple', linewidth=1.5, alpha=0.7)
    ax14.set_xlabel('Time (s)')
    ax14.set_ylabel('Wind Direction (deg)')
    ax14.set_title('WIND DIRECTION (0 deg = East)')
    ax14.grid(True, alpha=0.3)
    ax14.set_ylim([-180, 180])
    
    # 10. MOTOR THRUSTS IN NEWTONS
    ax15 = plt.subplot(4, 4, 15)
    ax15.plot(time_array, motors[:, 0], 'b-', alpha=0.7, linewidth=1)
    ax15.plot(time_array, motors[:, 1], 'g-', alpha=0.7, linewidth=1)
    ax15.plot(time_array, motors[:, 2], 'r-', alpha=0.7, linewidth=1)
    ax15.plot(time_array, motors[:, 3], 'orange', alpha=0.7, linewidth=1)
    ax15.axhline(y=drone.hover_thrust, color='k', linestyle='--', linewidth=2, label=f'Hover ({drone.hover_thrust:.2f}N)')
    ax15.set_xlabel('Time (s)')
    ax15.set_ylabel('Motor Thrust (N)')
    ax15.set_title('MOTOR THRUSTS vs TIME')
    ax15.grid(True, alpha=0.3)
    ax15.legend(loc='upper right', fontsize=6)
    
    # 11. 3D TRAJECTORY
    ax16 = plt.subplot(4, 4, 16, projection='3d')
    
    ax16.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
             'b-', alpha=0.7, linewidth=1)
    
    ax16.scatter(0, 0, 0, c='red', s=100, marker='*', label='Target (0,0,0)')
    
    ax16.set_xlabel('X (m)')
    ax16.set_ylabel('Y (m)')
    ax16.set_zlabel('Z (m)')
    ax16.set_title('3D TRAJECTORY')
    ax16.legend()
    ax16.grid(True, alpha=0.3)
    
    ax16.set_xlim([-0.5, 0.5])
    ax16.set_ylim([-0.5, 0.5])
    ax16.set_zlim([-0.3, 0.3])
    
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
    
    print(f"\nFinal Position: [{final_pos[0]:.4f}, {final_pos[1]:.4f}, {final_pos[2]:.4f}] m")
    print(f"Final Distance from Origin: {final_dist:.4f} m")
    print(f"Average Distance: {avg_dist:.4f} m")
    print(f"Maximum Distance: {max_dist:.4f} m")
    
    success_x = abs(final_pos[0]) < 0.1
    success_y = abs(final_pos[1]) < 0.1
    success_z = abs(final_pos[2]) < 0.1
    
    print(f"\nAxis Performance:")
    print(f"  X-axis: {'PASS' if success_x else 'FAIL'} Final: {final_pos[0]:.3f}m (target: 0m)")
    print(f"  Y-axis: {'PASS' if success_y else 'FAIL'} Final: {final_pos[1]:.3f}m (target: 0m)")
    print(f"  Z-axis: {'PASS' if success_z else 'FAIL'} Final: {final_pos[2]:.3f}m (target: 0m)")
    
    if success_x and success_y and success_z:
        print("\nSUCCESS! All axes maintain origin despite wind!")
    else:
        print("\nSome axes need improvement")
    
    print(f"\nMaximum Wind Encountered: {np.max(horizontal_wind):.1f} km/h")
    print(f"Drone Thrust/Weight Ratio: {4*drone.max_thrust/drone.weight:.0f}:1")

if __name__ == "__main__":
    main()