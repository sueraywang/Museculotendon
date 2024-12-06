import time
import glfw
from SimulatorNN import Simulator
from Renderer import OpenGLRenderer
from Plotter import SimulationPlotter
from Physics import DT

def main():
    simulator = Simulator()
    renderer = OpenGLRenderer(simulator)
    plotter = SimulationPlotter()

    while not renderer.should_close():
        glfw.poll_events()
        simulator.step()
        renderer.render()
        plotter.update(simulator, DT)
        time.sleep(DT)

    renderer.cleanup()

if __name__ == "__main__":
    main()