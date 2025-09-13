import time
from MuscleNN3D import Simulator
from Renderer import Renderer
from Plotter import SimulationPlotter
from Physics import *

def main():
    if RENDER_MT:
        l_M = find_initial_fiber_equilibrium(INITIAL_LENGTH, 0.5)
    else:
        l_M = INITIAL_LENGTH
    
    # Initialize the simulator
    simulator = Simulator(l_M)
    
    # Create the renderer with the new ModernOpenGLRenderer
    renderer = Renderer(simulator, cam_distance=3.0, MT=RENDER_MT)
    
    # Create a plotter for data visualization
    plotter = SimulationPlotter(MT=RENDER_MT)
    
    t = 0.0
    
    # Main simulation loop
    while t <= 10.0 and not renderer.should_close():
        # Process input
        renderer.process_input()
        
        # Calculate activation and pennation for plotting
        activation = np.sin(plotter.plot_time * 20) / 2.0 + 0.5
        
        # Render the scene
        renderer.render()
        
        # Update the plot
        plotter.update(simulator, DT, activation, simulator.penn)
        
        # Update the simulation
        simulator.step(activation)
        
        # Maintain consistent frame rate
        time.sleep(DT)
        t += DT

    # Export the simulation data
    #plotter.export_data("Results/precomputed_model_100steps.csv")
    
    # Clean up resources
    renderer.cleanup()

main()