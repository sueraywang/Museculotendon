import numpy as np
import matplotlib.pyplot as plt

# Physical constants
MASS = 0.1  # Mass of particle (kg)
REST_LENGTH = 1.0  # Natural spring length
SPRING_CONSTANT = 10  # Spring constant
DT = 0.001  # Time step
damping_factor = 1.0  # Damping factor for both methods
standard_damping_coeff = 0.5  # Coefficient for standard damping

# Objects' initial status - simplified 1D system
INITIAL_LENGTH = 0.5
FIXED_POS = 0.0  # Fixed particle at origin
FREE_POS = INITIAL_LENGTH  # Free particle initially displaced

# Simulation parameters
num_steps = 5000  # Total simulation steps

# Function for logistic damping
def logistic_damping(length_velocity):
    return 1.0 / (1.0 + np.exp(-length_velocity * damping_factor))

# Function for standard damping
def standard_damping(velocity):
    return standard_damping_coeff * velocity

# Simulate with logistic damping
def simulate_logistic_damping():
    # Initialize variables
    pos = FREE_POS
    vel = 0.0
    
    # Arrays to store results
    positions = np.zeros(num_steps)
    velocities = np.zeros(num_steps)
    spring_lengths = np.zeros(num_steps)
    times = np.zeros(num_steps)
    
    # Initial values
    positions[0] = pos
    spring_lengths[0] = abs(pos - FIXED_POS)
    
    # Main simulation loop - Symplectic Euler
    for i in range(1, num_steps):
        # Calculate current spring length and extension
        current_length = abs(pos - FIXED_POS)
        extension = current_length - REST_LENGTH
        
        # Direction (1D case: right is positive)
        direction = 1.0 if pos > FIXED_POS else -1.0 if pos < FIXED_POS else 0.0
        
        # Calculate spring length velocity
        if i > 1:
            length_velocity = (current_length - spring_lengths[i-1]) / DT
        else:
            length_velocity = 0.0
        
        # Apply logistic damping
        damping = logistic_damping(length_velocity)
        
        # Calculate spring force: F = -k*x*damping
        spring_force = -SPRING_CONSTANT * extension * damping * direction
        
        # Acceleration
        acc = spring_force / MASS
        
        # Symplectic Euler integration
        vel = vel + acc * DT
        pos = pos + vel * DT
        
        # Store results
        positions[i] = pos
        velocities[i] = vel
        spring_lengths[i] = current_length
        times[i] = i * DT
    
    return times, spring_lengths, velocities

# Simulate with standard damping
def simulate_standard_damping():
    # Initialize variables
    pos = FREE_POS
    vel = 0.0
    
    # Arrays to store results
    positions = np.zeros(num_steps)
    velocities = np.zeros(num_steps)
    spring_lengths = np.zeros(num_steps)
    times = np.zeros(num_steps)
    
    # Initial values
    positions[0] = pos
    spring_lengths[0] = abs(pos - FIXED_POS)
    
    # Main simulation loop - Symplectic Euler
    for i in range(1, num_steps):
        # Calculate current spring length and extension
        current_length = abs(pos - FIXED_POS)
        extension = current_length - REST_LENGTH
        
        # Direction (1D case: right is positive)
        direction = 1.0 if pos > FIXED_POS else -1.0 if pos < FIXED_POS else 0.0
        
        # Calculate spring force: F = -k*x - c*v
        spring_force = -SPRING_CONSTANT * extension * direction
        damping_force = -standard_damping(vel)  # Standard velocity damping
        
        # Total force
        total_force = spring_force + damping_force
        
        # Acceleration
        acc = total_force / MASS
        
        # Symplectic Euler integration
        vel = vel + acc * DT
        pos = pos + vel * DT
        
        # Store results
        positions[i] = pos
        velocities[i] = vel
        spring_lengths[i] = current_length
        times[i] = i * DT
    
    return times, spring_lengths, velocities

# Run both simulations
times_logistic, lengths_logistic, velocities_logistic = simulate_logistic_damping()
times_standard, lengths_standard, velocities_standard = simulate_standard_damping()

# Create comparison plots
plt.figure(figsize=(12, 12))

# Plot 1: Spring Length vs Time - Both Methods
plt.subplot(3, 1, 1)
plt.plot(times_logistic, lengths_logistic, 'b-', linewidth=1.5, label='Logistic Damping')
plt.plot(times_standard, lengths_standard, 'r--', linewidth=1.5, label='Standard Damping')
plt.axhline(y=REST_LENGTH, color='k', linestyle=':', label='Natural Length')
plt.xlabel('Time (s)')
plt.ylabel('Spring Length (m)')
plt.grid(True)
plt.legend()
plt.title('Spring Length vs Time Comparison')

# Plot 2: Zoomed in comparison of first few oscillations
plt.subplot(3, 1, 2)
zoom_steps = 1000  # Show first 1000 steps
plt.plot(times_logistic[:zoom_steps], lengths_logistic[:zoom_steps], 'b-', linewidth=1.5, label='Logistic Damping')
plt.plot(times_standard[:zoom_steps], lengths_standard[:zoom_steps], 'r--', linewidth=1.5, label='Standard Damping')
plt.axhline(y=REST_LENGTH, color='k', linestyle=':', label='Natural Length')
plt.xlabel('Time (s)')
plt.ylabel('Spring Length (m)')
plt.grid(True)
plt.legend()
plt.title('Spring Length vs Time (Zoomed)')

# Plot 3: Velocity vs Time - Both Methods
plt.subplot(3, 1, 3)
plt.plot(times_logistic[:zoom_steps], velocities_logistic[:zoom_steps], 'b-', linewidth=1.5, label='Logistic Damping')
plt.plot(times_standard[:zoom_steps], velocities_standard[:zoom_steps], 'r--', linewidth=1.5, label='Standard Damping')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.grid(True)
plt.legend()
plt.title('Velocity vs Time Comparison')

plt.tight_layout()
plt.show()

# Calculate decay characteristics
def calculate_decay_characteristics(times, lengths):
    # Find peaks for envelope
    peaks = []
    for i in range(1, len(lengths)-1):
        if lengths[i] > lengths[i-1] and lengths[i] > lengths[i+1]:
            peaks.append((times[i], lengths[i]))
    
    if len(peaks) < 3:
        return "Not enough peaks to calculate decay rate"
    
    # Calculate decay rate from first few peaks
    peak_times = [p[0] for p in peaks[:5]] if len(peaks) >= 5 else [p[0] for p in peaks]
    peak_heights = [p[1] for p in peaks[:5]] if len(peaks) >= 5 else [p[1] for p in peaks]
    
    # Calculate periods
    periods = [peak_times[i+1] - peak_times[i] for i in range(len(peak_times)-1)]
    avg_period = np.mean(periods) if periods else 0
    
    # Calculate amplitude ratios for decay rate
    amplitude_ratios = [peak_heights[i]/peak_heights[i+1] for i in range(len(peak_heights)-1)]
    avg_ratio = np.mean(amplitude_ratios) if amplitude_ratios else 0
    
    return {
        "average_period": avg_period,
        "amplitude_decay_ratio": avg_ratio,
        "number_of_peaks": len(peaks)
    }

# Calculate and print characteristics
logistic_characteristics = calculate_decay_characteristics(times_logistic, lengths_logistic)
standard_characteristics = calculate_decay_characteristics(times_standard, lengths_standard)

print("Logistic Damping Characteristics:")
print(logistic_characteristics)
print("\nStandard Damping Characteristics:")
print(standard_characteristics)

# Plot decay envelopes
def plot_decay_envelopes():
    plt.figure(figsize=(12, 6))
    
    # Find peaks for both methods
    logistic_peaks = []
    standard_peaks = []
    
    for i in range(1, len(lengths_logistic)-1):
        if lengths_logistic[i] > lengths_logistic[i-1] and lengths_logistic[i] > lengths_logistic[i+1]:
            logistic_peaks.append((times_logistic[i], lengths_logistic[i]))
    
    for i in range(1, len(lengths_standard)-1):
        if lengths_standard[i] > lengths_standard[i-1] and lengths_standard[i] > lengths_standard[i+1]:
            standard_peaks.append((times_standard[i], lengths_standard[i]))
    
    # Plot spring lengths
    plt.plot(times_logistic, lengths_logistic, 'b-', alpha=0.5, label='Logistic Damping')
    plt.plot(times_standard, lengths_standard, 'r-', alpha=0.5, label='Standard Damping')
    
    # Plot peak envelopes
    if logistic_peaks:
        peak_times_l = [p[0] for p in logistic_peaks]
        peak_heights_l = [p[1] for p in logistic_peaks]
        plt.plot(peak_times_l, peak_heights_l, 'bo-', label='Logistic Peaks')
    
    if standard_peaks:
        peak_times_s = [p[0] for p in standard_peaks]
        peak_heights_s = [p[1] for p in standard_peaks]
        plt.plot(peak_times_s, peak_heights_s, 'ro-', label='Standard Peaks')
    
    plt.axhline(y=REST_LENGTH, color='k', linestyle=':', label='Natural Length')
    plt.xlabel('Time (s)')
    plt.ylabel('Spring Length (m)')
    plt.grid(True)
    plt.legend()
    plt.title('Decay Envelope Comparison')
    
    plt.tight_layout()
    plt.show()

plot_decay_envelopes()