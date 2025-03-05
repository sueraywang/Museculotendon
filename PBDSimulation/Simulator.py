import time
import glfw
import numpy as np
from MuscleNN import Simulator
from Renderer import OpenGLRenderer
from Plotter import SimulationPlotter
from Physics import DT

def main():
    simulator = Simulator()
    renderer = OpenGLRenderer(simulator)
    plotter = SimulationPlotter()

    while not renderer.should_close():
        glfw.poll_events()
        activation = np.sin(plotter.plot_time * 20) / 2.0 + 0.5
        
        simulator.step(activation)
        renderer.render()
        plotter.update(simulator, DT, activation)
        time.sleep(DT)

    renderer.cleanup()

if __name__ == "__main__":
    main()