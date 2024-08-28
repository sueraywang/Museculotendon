import numpy as np
import matplotlib.pyplot as plt
from SmoothSegmentedFunction import *

class CurveBase:
    m_curve = None
    def __init__(self, curve):
        self.m_curve = curve
    
    def calcValue(self, x):
        return self.m_curve.calcValue(x)
    
    def calcDerivative(self, x, order):
        return self.m_curve.calcDerivative(x, order)
    
    def calcValDeriv(self, x):
        return self.m_curve.calcValDeriv(x)
    
    @staticmethod
    def test(curve, cname, x):
        """
        Test the curve class and plot its value and derivatives.

        Args:
            curve: An instance of a class that has `calc_val_deriv` method.
            cname: The name of the curve for plotting title.
            x: Array of x values for testing.
        """
        y0 = np.zeros_like(x)
        y1 = np.zeros_like(x)
        y2 = np.zeros_like(x)
        
        for i in range(len(x)):
            y0 = curve.calcValDeriv(x)[0]
            y1 = curve.calcValDeriv(x)[1]
            y2 = curve.calcValDeriv(x)[2]
        
        # Plot results
        plt.figure(figsize=(10, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(x, y0, linewidth=2)
        plt.title(cname)
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot(x, y1, linewidth=2)
        plt.title('1st derivative')
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.plot(x, y2, linewidth=2)
        plt.title('2nd derivative')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        return y0,y1,y2
