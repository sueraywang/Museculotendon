import time
import glfw
import numpy as np
from SpringDamped import Simulator
from Renderer import OpenGLRenderer
from Plotter import SimulationPlotter
from Physics import DT

def main():
    simulator = Simulator()
    renderer = OpenGLRenderer(simulator)
    plotter = SimulationPlotter(muscle=False)

    while not renderer.should_close():
        glfw.poll_events()
        activation = np.sin(plotter.plot_time * 20) / 2.0 + 0.5
        pennation = (np.cos(plotter.plot_time * 10) / 2.0 + 0.5) * 0.6
        
        simulator.step()
        renderer.render()
        plotter.update(simulator, DT, activation, pennation)
        time.sleep(DT)

    renderer.cleanup()

if __name__ == "__main__":
    main()