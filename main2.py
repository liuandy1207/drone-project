# test_wind.py

from drone_forces import OUWindGenerator
import numpy as np
import matplotlib.pyplot as plt

# Initialize the wind generator
dt = 0.01
duration = 10  # seconds
wind_gen = OUWindGenerator(dt)

# Arrays to store wind data
time_array = np.arange(0, duration, dt)
wind_history = []

# Generate wind over time
for t in time_array:
    wind = wind_gen.update()
    wind_history.append(wind.copy())

wind_history = np.array(wind_history)

# Plot the wind components
plt.figure(figsize=(10, 6))
plt.plot(time_array, wind_history[:,0], label='Wind X')
plt.plot(time_array, wind_history[:,1], label='Wind Y')
plt.plot(time_array, wind_history[:,2], label='Wind Z')
plt.xlabel('Time (s)')
plt.ylabel('Wind Speed (m/s)')
plt.title('Ornstein-Uhlenbeck Wind Simulation')
plt.legend()
plt.grid(True)
plt.show()
