import time
#from mutipleCSim import Simulator
from Simulator import Simulator
from Renderer import Renderer
from Plotter import SimulationPlotter
from Physics import *

def main():
    if RENDER_MT:
        l_M = find_initial_fiber_equilibrium(INITIAL_LENGTH, 0.5)
    else:
        l_M = INITIAL_LENGTH
    
    # Initialize the simulator
    nn_model = MLP(input_size=model_params['input_size'], hidden_size=model_params['num_width'], 
            output_size=model_params['output_size'], num_layers=model_params['num_layer'], 
            activation=model_params['activation_func'])
    checkpoint = torch.load(os.path.join('TrainedModels/Muscles', model_params['model_name']), weights_only=True)
    nn_model.load_state_dict(checkpoint['model_state_dict'])
    nn_model.eval()
    penn_model = model_params['includes_penn']
    damp_model = model_params['includes_damp']
    
    simulator = Simulator(l_M, penn_model, damp_model, nn_model)
    #simulator = Simulator(l_M)
    
    # Create the renderer with the new ModernOpenGLRenderer
    renderer = Renderer(simulator, cam_distance=3.0, MT=RENDER_MT)
    
    # Create a plotter for data visualization
    plotter = SimulationPlotter(MT=RENDER_MT)
    
    t = 0.0
    
    # Main simulation loop
    while t <= 5.0 and not renderer.should_close():
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
    base_name = os.path.splitext(os.path.basename(model_params['model_name']))[0]
    csv_name = base_name + ".csv"
    #plotter.export_data(os.path.join('Results', csv_name))
    
    # Clean up resources
    renderer.cleanup()

main()