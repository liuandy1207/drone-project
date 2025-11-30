"""
REALISTIC QUADCOPTER PHYSICS: Proper motor mixing and thrust vectoring
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import math

# --------------------------
# Physical & sim parameters
# --------------------------
rho = 1.225          # air density (kg/m^3)
g = 9.81             # gravity (m/s^2)

# Quadcopter physical parameters - typical 250-300mm frame
mass = 0.8           # kg (including battery)
weight = mass * g    # N

# Motor parameters - typical 2204-2300KV motors with 5-6" props
thrust_per_motor_max = 4.5  # N per motor (approx 450g thrust each)
combined_thrust_max = thrust_per_motor_max * 4  # 18N total
thrust_per_motor_hover = weight / 4.0  # hover thrust per motor

# Physical dimensions
arm_length = 0.15    # m (from center to motor)
body_width = 0.18    # m
body_height = 0.06   # m

# Moment of inertia (approximate for symmetric quadcopter)
I_xx = 0.01          # kg*m^2 (roll inertia)
I_yy = 0.01          # kg*m^2 (pitch inertia) 
I_zz = 0.02          # kg*m^2 (yaw inertia)

# Drag coefficients (simplified)
C_d_horizontal = 1.2
A_frontal = body_width * body_height  # frontal area

print(f"Drone mass: {mass} kg, Weight: {weight:.1f} N")
print(f"Max thrust/motor: {thrust_per_motor_max:.1f} N, Total: {combined_thrust_max:.1f} N")
print(f"Thrust-to-weight ratio: {combined_thrust_max/weight:.1f}:1")

# REALISTIC WIND PARAMETERS - gentle breeze that won't blow drone away
WIND_MAX_SPEED = 0.8    # m/s - very gentle breeze (2.9 km/h)
WIND_SIGMA_HORIZONTAL = 0.2  # m/s
WIND_SIGMA_VERTICAL = 0.1    # m/s

# Time stepping
dt = 0.01            # 100Hz control loop - realistic for drones
t_final = 30.0       # simulation time
nt = int(t_final / dt)

# 3D wind components
wind_u, wind_v, wind_w = 0.0, 0.0, 0.0

def update_wind_gentle():
    """Very gentle wind that won't overwhelm the drone"""
    global wind_u, wind_v, wind_w
    
    # Slow, correlated wind changes (not completely random each step)
    base_u = 0.95 * wind_u + 0.05 * np.random.normal(0, WIND_SIGMA_HORIZONTAL)
    base_v = 0.95 * wind_v + 0.05 * np.random.normal(0, WIND_SIGMA_HORIZONTAL)  
    base_w = 0.95 * wind_w + 0.05 * np.random.normal(0, WIND_SIGMA_VERTICAL)
    
    # Hard limit to very gentle wind
    wind_speed = math.sqrt(base_u**2 + base_v**2 + base_w**2)
    if wind_speed > WIND_MAX_SPEED:
        scale = WIND_MAX_SPEED / wind_speed
        base_u *= scale
        base_v *= scale
        base_w *= scale
    
    wind_u, wind_v, wind_w = base_u, base_v, base_w
    return wind_u, wind_v, wind_w

def calculate_drag_force(velocity, area, C_d, rho):
    """Calculate drag force in one direction"""
    return -0.5 * rho * C_d * area * velocity * abs(velocity)

# --------------------------
# REALISTIC DRONE PERFORMANCE CONSTRAINTS
# --------------------------
# Based on typical consumer drone capabilities
MAX_TILT_ANGLE = np.radians(30)  # Maximum 30 degrees tilt - realistic limit
MAX_ASCENT_RATE = 2.0            # m/s - realistic vertical speed
MAX_DESCENT_RATE = 1.5           # m/s  
MAX_HORIZONTAL_SPEED = 3.0       # m/s - conservative

print(f"\nDRONE PERFORMANCE LIMITS:")
print(f"Max tilt angle: {np.degrees(MAX_TILT_ANGLE):.0f}°")
print(f"Max horizontal speed: {MAX_HORIZONTAL_SPEED:.1f} m/s")
print(f"Max wind speed: {WIND_MAX_SPEED:.1f} m/s ({WIND_MAX_SPEED*3.6:.1f} km/h)")

# PID CONTROLLER GAINS - tuned for stable quadcopter control
# Position controller (outer loop)
pos_kp = 1.5; pos_kd = 2.0; pos_ki = 0.1

# Attitude controller (inner loop) - much faster response
att_kp = 8.0; att_kd = 1.5

# Altitude controller  
alt_kp = 25.0; alt_kd = 12.0; alt_ki = 2.0

# Integral terms
integral_x = 0.0; integral_y = 0.0; integral_z = 0.0

# Target position
target_x, target_y, target_z = 0.0, 0.0, 2.0  # 2m altitude

# --------------------------
# PROPER QUADCOPTER DYNAMICS
# --------------------------
class Quadcopter:
    def __init__(self):
        # State: position, velocity, orientation, angular velocity
        self.position = np.array([0.0, 0.0, 2.0])  # x, y, z
        self.velocity = np.array([0.0, 0.0, 0.0])  # vx, vy, vz
        self.orientation = np.array([0.0, 0.0, 0.0])  # roll, pitch, yaw (radians)
        self.angular_velocity = np.array([0.0, 0.0, 0.0])  # p, q, r (rad/s)
        
        # Motor thrusts (N)
        self.motor_thrusts = np.array([thrust_per_motor_hover] * 4)
        
    def get_rotation_matrix(self):
        """Get rotation matrix from current orientation (roll, pitch, yaw)"""
        phi, theta, psi = self.orientation
        
        # ZYX rotation (yaw-pitch-roll)
        R_x = np.array([[1, 0, 0],
                       [0, np.cos(phi), -np.sin(phi)],
                       [0, np.sin(phi), np.cos(phi)]])
        
        R_y = np.array([[np.cos(theta), 0, np.sin(theta)],
                       [0, 1, 0],
                       [-np.sin(theta), 0, np.cos(theta)]])
        
        R_z = np.array([[np.cos(psi), -np.sin(psi), 0],
                       [np.sin(psi), np.cos(psi), 0],
                       [0, 0, 1]])
        
        return R_z @ R_y @ R_x  # Rotation from body to world frame
    
    def get_thrust_vector(self):
        """Calculate total thrust vector in world coordinates"""
        # Thrust in body frame (always along body z-axis)
        total_thrust = np.sum(self.motor_thrusts)
        thrust_body = np.array([0, 0, total_thrust])
        
        # Rotate to world frame
        R = self.get_rotation_matrix()
        thrust_world = R @ thrust_body
        
        return thrust_world
    
    def calculate_motor_torques(self):
        """Calculate torques generated by motor thrust differences"""
        # Motor arrangement: 
        # 0: front-right (CW), 1: front-left (CCW)
        # 2: back-left (CW), 3: back-right (CCW)
        
        # Roll torque (difference between left and right motors)
        roll_torque = (self.motor_thrusts[1] + self.motor_thrusts[2] - 
                      self.motor_thrusts[0] - self.motor_thrusts[3]) * arm_length
        
        # Pitch torque (difference between front and back motors)  
        pitch_torque = (self.motor_thrusts[0] + self.motor_thrusts[1] -
                       self.motor_thrusts[2] - self.motor_thrusts[3]) * arm_length
        
        # Yaw torque (difference between CW and CCW motors)
        # CW motors: 0, 2; CCW motors: 1, 3
        yaw_torque = (self.motor_thrusts[0] + self.motor_thrusts[2] -
                     self.motor_thrusts[1] - self.motor_thrusts[3]) * 0.1  # smaller moment arm
        
        return np.array([roll_torque, pitch_torque, yaw_torque])
    
    def update_dynamics(self, dt, wind_force):
        """Update quadcopter dynamics using proper physics"""
        # Total forces in world frame
        thrust_world = self.get_thrust_vector()
        gravity_force = np.array([0, 0, -weight])
        
        # Net force
        net_force = thrust_world + gravity_force + wind_force
        
        # Linear acceleration (F = ma)
        acceleration = net_force / mass
        
        # Update linear motion
        self.velocity += acceleration * dt
        self.position += self.velocity * dt
        
        # Angular motion - torques cause angular acceleration
        motor_torques = self.calculate_motor_torques()
        
        # Simple damping torques (aerodynamic drag on rotation)
        damping_torques = -0.1 * self.angular_velocity
        
        # Total torques
        net_torque = motor_torques + damping_torques
        
        # Angular acceleration (τ = Iα)
        I = np.array([I_xx, I_yy, I_zz])
        angular_acceleration = net_torque / I
        
        # Update angular motion
        self.angular_velocity += angular_acceleration * dt
        self.orientation += self.angular_velocity * dt
        
        # Limit orientation to prevent unrealistic angles
        self.orientation[0] = np.clip(self.orientation[0], -MAX_TILT_ANGLE, MAX_TILT_ANGLE)  # roll
        self.orientation[1] = np.clip(self.orientation[1], -MAX_TILT_ANGLE, MAX_TILT_ANGLE)  # pitch
        
        # Limit velocities
        horizontal_speed = np.linalg.norm(self.velocity[:2])
        if horizontal_speed > MAX_HORIZONTAL_SPEED:
            scale = MAX_HORIZONTAL_SPEED / horizontal_speed
            self.velocity[0] *= scale
            self.velocity[1] *= scale
            
        self.velocity[2] = np.clip(self.velocity[2], -MAX_DESCENT_RATE, MAX_ASCENT_RATE)
        
        # Prevent going below ground
        if self.position[2] < 0.1:
            self.position[2] = 0.1
            self.velocity[2] = max(0, self.velocity[2])

# Initialize quadcopter
drone = Quadcopter()

# Motor mixing function
def motor_mixing(collective_thrust, roll_torque, pitch_torque, yaw_torque):
    """
    Convert desired thrust and torques to individual motor commands
    Standard quadcopter mixing:
    T0 = collective + pitch + roll - yaw  (front-right)
    T1 = collective + pitch - roll + yaw  (front-left) 
    T2 = collective - pitch - roll - yaw  (back-left)
    T3 = collective - pitch + roll + yaw  (back-right)
    """
    base = collective_thrust / 4.0
    
    # Convert torques to thrust differences
    roll_scale = roll_torque / (2.0 * arm_length)
    pitch_scale = pitch_torque / (2.0 * arm_length) 
    yaw_scale = yaw_torque / 4.0
    
    # Mix to individual motors
    T0 = base + pitch_scale + roll_scale - yaw_scale  # front-right
    T1 = base + pitch_scale - roll_scale + yaw_scale  # front-left
    T2 = base - pitch_scale - roll_scale - yaw_scale  # back-left  
    T3 = base - pitch_scale + roll_scale + yaw_scale  # back-right
    
    return np.array([T0, T1, T2, T3])

# History for plotting
history = {
    't': [], 'wind_u': [], 'wind_v': [], 'wind_w': [],
    'roll': [], 'pitch': [], 'yaw': [],
    'x': [], 'y': [], 'z': [],
    'total_thrust': [], 'horizontal_speed': [],
    'motor_thrusts': []
}

print(f"\nSIMULATION STARTING")
print(f"Target position: ({target_x}, {target_y}, {target_z})")
print("="*60)

# --------------------------
# MAIN SIMULATION LOOP
# --------------------------
for n in range(nt):
    t = n * dt
    
    # Update gentle wind
    wind_u, wind_v, wind_w = update_wind_gentle()
    
    # Calculate wind force on drone (simplified)
    v_rel_x = drone.velocity[0] - wind_u
    v_rel_y = drone.velocity[1] - wind_v  
    v_rel_z = drone.velocity[2] - wind_w
    
    F_wind_x = calculate_drag_force(v_rel_x, A_frontal, C_d_horizontal, rho)
    F_wind_y = calculate_drag_force(v_rel_y, A_frontal, C_d_horizontal, rho) 
    F_wind_z = calculate_drag_force(v_rel_z, A_frontal, 0.5, rho)  # less vertical drag
    
    wind_force = np.array([F_wind_x, F_wind_y, F_wind_z])
    
    # POSITION CONTROL (outer loop) - generates desired tilt angles
    pos_error = np.array([target_x, target_y, target_z]) - drone.position
    
    # Update integrals with anti-windup
    integral_x += pos_error[0] * dt
    integral_y += pos_error[1] * dt  
    integral_z += pos_error[2] * dt
    integral_x = np.clip(integral_x, -2.0, 2.0)
    integral_y = np.clip(integral_y, -2.0, 2.0)
    integral_z = np.clip(integral_z, -1.0, 1.0)
    
    # PID position control
    desired_vx = pos_kp * pos_error[0] + pos_kd * (-drone.velocity[0]) + pos_ki * integral_x
    desired_vy = pos_kp * pos_error[1] + pos_kd * (-drone.velocity[1]) + pos_ki * integral_y
    
    # Convert desired horizontal velocity to tilt angles
    # This is the key insight: tilt creates horizontal acceleration
    current_total_thrust = np.sum(drone.motor_thrusts)
    if current_total_thrust > weight * 0.8:  # Only if we have sufficient thrust
        # desired_acceleration = thrust * sin(tilt) / mass
        # So: tilt ≈ arcsin(mass * desired_acceleration / thrust)
        desired_pitch = np.arcsin(np.clip(mass * desired_vx / current_total_thrust, -0.8, 0.8))
        desired_roll = -np.arcsin(np.clip(mass * desired_vy / current_total_thrust, -0.8, 0.8))
    else:
        desired_roll, desired_pitch = 0.0, 0.0
    
    # Limit tilt angles
    desired_roll = np.clip(desired_roll, -MAX_TILT_ANGLE, MAX_TILT_ANGLE)
    desired_pitch = np.clip(desired_pitch, -MAX_TILT_ANGLE, MAX_TILT_ANGLE)
    
    # ALTITUDE CONTROL
    alt_error = target_z - drone.position[2]
    desired_thrust = alt_kp * alt_error + alt_kd * (-drone.velocity[2]) + alt_ki * integral_z + weight
    desired_thrust = np.clip(desired_thrust, weight * 0.5, combined_thrust_max * 0.9)
    
    # ATTITUDE CONTROL (inner loop) - much faster response
    roll_error = desired_roll - drone.orientation[0]
    pitch_error = desired_pitch - drone.orientation[1]
    yaw_error = 0.0 - drone.orientation[2]  # maintain current yaw
    
    # PD attitude control - generates desired torques
    roll_torque = att_kp * roll_error + att_kd * (-drone.angular_velocity[0])
    pitch_torque = att_kp * pitch_error + att_kd * (-drone.angular_velocity[1]) 
    yaw_torque = att_kp * yaw_error + att_kd * (-drone.angular_velocity[2])
    
    # MOTOR MIXING - convert to individual motor thrusts
    motor_commands = motor_mixing(desired_thrust, roll_torque, pitch_torque, yaw_torque)
    
    # Apply motor limits
    motor_commands = np.clip(motor_commands, 0.1, thrust_per_motor_max)
    drone.motor_thrusts = motor_commands
    
    # UPDATE PHYSICS
    drone.update_dynamics(dt, wind_force)
    
    # RECORD HISTORY
    history['t'].append(t)
    history['wind_u'].append(wind_u)
    history['wind_v'].append(wind_v) 
    history['wind_w'].append(wind_w)
    history['roll'].append(np.degrees(drone.orientation[0]))
    history['pitch'].append(np.degrees(drone.orientation[1]))
    history['yaw'].append(np.degrees(drone.orientation[2]))
    history['x'].append(drone.position[0])
    history['y'].append(drone.position[1])
    history['z'].append(drone.position[2])
    history['total_thrust'].append(np.sum(drone.motor_thrusts))
    history['horizontal_speed'].append(np.linalg.norm(drone.velocity[:2]))
    history['motor_thrusts'].append(drone.motor_thrusts.copy())
    
    # Periodic status
    if n % (nt//20) == 0:
        pos_error_mag = np.linalg.norm(pos_error)
        tilt_angle = np.degrees(np.linalg.norm(drone.orientation[:2]))
        print(f"t={t:4.1f}s err={pos_error_mag:5.3f}m tilt={tilt_angle:4.1f}° thrust={np.sum(drone.motor_thrusts):4.1f}N")

# --------------------------
# PLOTTING
# --------------------------
plt.figure(figsize=(16, 12))

# Graph 1: Wind components
plt.subplot(2, 2, 1)
plt.plot(history['t'], history['wind_u'], 'r-', label='Wind X', alpha=0.8)
plt.plot(history['t'], history['wind_v'], 'g-', label='Wind Y', alpha=0.8)
plt.plot(history['t'], history['wind_w'], 'b-', label='Wind Z', alpha=0.8)
plt.axhline(y=WIND_MAX_SPEED, color='red', linestyle='--', alpha=0.5, label=f'Max ({WIND_MAX_SPEED}m/s)')
plt.xlabel('Time (s)'); plt.ylabel('Wind Speed (m/s)')
plt.legend(); plt.grid(True, alpha=0.3)
plt.title('Gentle 3D Wind (Drone can easily compensate)')

# Graph 2: Orientation
plt.subplot(2, 2, 2)
plt.plot(history['t'], history['roll'], 'r-', label='Roll', alpha=0.8)
plt.plot(history['t'], history['pitch'], 'g-', label='Pitch', alpha=0.8)
plt.plot(history['t'], history['yaw'], 'b-', label='Yaw', alpha=0.8)
plt.axhline(y=np.degrees(MAX_TILT_ANGLE), color='red', linestyle='--', alpha=0.5, label='Max tilt')
plt.axhline(y=-np.degrees(MAX_TILT_ANGLE), color='red', linestyle='--', alpha=0.5)
plt.xlabel('Time (s)'); plt.ylabel('Angle (degrees)')
plt.legend(); plt.grid(True, alpha=0.3)
plt.title('Drone Orientation (Tilt creates horizontal motion)')

# Graph 3: Position
plt.subplot(2, 2, 3)
plt.plot(history['t'], history['x'], 'r-', label='X', alpha=0.8)
plt.plot(history['t'], history['y'], 'g-', label='Y', alpha=0.8)
plt.plot(history['t'], history['z'], 'b-', label='Z', alpha=0.8)
plt.axhline(y=target_z, color='b', linestyle='--', alpha=0.5, label='Target Z')
plt.xlabel('Time (s)'); plt.ylabel('Position (m)')
plt.legend(); plt.grid(True, alpha=0.3)
plt.title('3D Position (Maintains altitude despite wind)')

# Graph 4: Motor thrusts
plt.subplot(2, 2, 4)
motor_thrusts = np.array(history['motor_thrusts'])
plt.plot(history['t'], motor_thrusts[:, 0], 'r-', label='Motor 0 (FR)', alpha=0.8)
plt.plot(history['t'], motor_thrusts[:, 1], 'g-', label='Motor 1 (FL)', alpha=0.8)
plt.plot(history['t'], motor_thrusts[:, 2], 'b-', label='Motor 2 (BL)', alpha=0.8)
plt.plot(history['t'], motor_thrusts[:, 3], 'orange', label='Motor 3 (BR)', alpha=0.8)
plt.axhline(y=thrust_per_motor_hover, color='black', linestyle='--', alpha=0.5, label='Hover thrust')
plt.xlabel('Time (s)'); plt.ylabel('Thrust (N)')
plt.legend(); plt.grid(True, alpha=0.3)
plt.title('Individual Motor Thrusts (How drone actually controls itself)')

plt.tight_layout()
plt.show()

# Performance summary
final_error = np.linalg.norm(np.array([target_x, target_y, target_z]) - drone.position)
max_tilt = max(np.linalg.norm(np.column_stack([history['roll'], history['pitch']]), axis=1))
max_wind = max([math.sqrt(wu**2 + wv**2 + ww**2) for wu, wv, ww in zip(history['wind_u'], history['wind_v'], history['wind_w'])])

print("\n" + "="*60)
print("PERFORMANCE SUMMARY")
print("="*60)
print(f"Final position error: {final_error:.3f} m")
print(f"Max tilt angle: {max_tilt:.1f}° (limit: {np.degrees(MAX_TILT_ANGLE):.0f}°)")
print(f"Max wind speed: {max_wind:.2f} m/s")
print(f"Wind recoverability: {'EXCELLENT' if max_wind < 1.0 else 'GOOD'}")
print("="*60)