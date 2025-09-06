import time
from SpringDamped import Simulator
from Renderer import Renderer
from Plotter import SimulationPlotter
from Physics import DT

def main():
    # Initialize the simulator
    simulator = Simulator()
    
    # Create the renderer with the new ModernOpenGLRenderer
    renderer = Renderer(simulator, cam_distance=5.0)
    
    # Create a plotter for data visualization
    plotter = SimulationPlotter()
    
    # Main simulation loop
    while not renderer.should_close():
        # Process input
        renderer.process_input()
        
        # Render the scene
        renderer.render()
        
        # Update the simulation
        simulator.step()
        
        # Update the plot
        plotter.update(simulator, DT)
        
        # Maintain consistent frame rate
        time.sleep(DT)

    # Clean up resources
    renderer.cleanup()

if __name__ == "__main__":
    main()