import pandas as pd
import matplotlib.pyplot as plt

def plot_simulation_data(filename="xpbd_simulation_data.csv"):
    # Load the exported CSV
    df = pd.read_csv(filename)
    
    # Compute error
    df['error'] = df['xpbd_length'] - df['classic_length']
    
    # Plot the simulation curves
    plt.figure(figsize=(12, 5))
    
    # Plot lengths
    plt.subplot(1, 2, 1)
    plt.plot(df['time'], df['xpbd_length'], label='XPBD Length', linewidth=2)
    plt.plot(df['time'], df['classic_length'], label='Classic Length', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Length')
    plt.title('XPBD vs. Classic Length Over Time')
    plt.legend()
    plt.grid(True)

    # Plot absolute error
    plt.subplot(1, 2, 2)
    plt.plot(df['time'], df['error'], color='red')
    plt.xlabel('Time')
    plt.ylabel('Error')
    plt.title('Error Between XPBD and Classic Length')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("len_act_vel_penn_model.png")
    plt.show()

# Example usage:
plot_simulation_data("len_act_vel_penn_model.csv")
