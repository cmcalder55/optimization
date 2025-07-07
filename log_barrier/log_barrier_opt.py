#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created On:   2024/09/14
Last Revision: 0000/00/00

Log-Barrier Optimization.
'''

import inspect
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as opt

__author__= "Cameron Calder"
__maintainer__= "Cameron Calder"
__email__=""
__copyright__ = "(C)Copyright 2024-Present, Cameron Calder"
__license__=""
__version__= "0.0.0"


class LogBarrierOpt():
    """Log-Barrier Optimization main class.
    """
    def __init__(self, hx, fx, start, r="", t_stop="", r_range = (4, 6), t_stop_range = (0.001, 0.01)):     

        self.fx = fx
        self.hx = hx

        self.i = 0
            
        if not r or not t_stop:
            r, t_stop = self.tune_parameters(r_range, t_stop_range, start)

        self.result = self.getPoints(start, r, t_stop)       

    def plotContour(self, pts):
        """Plots (left) Optimization results on a contour plot illustrating the feasible
        region within the barrier formed by the contraints.  
        Plots (right) objective function value over iterations/

        :param list pts: Points calculated during optimization
        """
        
        n_pts = len(pts)
        # make contour plot of x1 vs. x2 from vertices in feasible set
        d = np.linspace(0.2, 1, 300)

        x1, x2 = np.meshgrid(d, d)
        f_x = list(map(lambda x: self.fx(x), zip(x1,x2)))

        # Calculate the objective function value for each point
        objective_values = [self.fx(pt) for pt in pts]

        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        contours = ax1.contour(x1, x2, f_x)
        
        # add a colorbar and labels
        ax1.set_title('f(x1,x2) Contour Plot and Feasible Region')
        ax1.set_xlabel('x1')
        ax1.set_ylabel('x2')

        extent=(x1.min(), x1.max(), x2.min(), x2.max())
        region = (x1 + x2, 1 - x1**2 - x2**2)
        region = ((region[0] >= 0) & (region[1] >= 0))
        ax1.imshow(region, interpolation='nearest', extent=extent, origin="lower", cmap="Greys", alpha=0.3)
        ax1.clabel(contours, inline=1, fontsize=10)
        ax1.plot(*zip(*pts), marker='o', markersize=4, label="Objective Function", color='tomato', markeredgecolor='r')

        x_final, y_final = pts[-1]
        print(f"Final Objective Function Value: ({x_final:.2f} , {y_final:.2f})\n")
        ax1.text(x_final+0.03, y_final, f'Final: ({x_final:.2f} , {y_final:.2f})', 
                 ha='left', va='bottom', bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.6))

        # Add direction using arrows
        for i in range( n_pts - 1 ):
            plot_pts = (
                        pts[i][0], pts[i][1],
                        pts[i+1][0] - pts[i][0], pts[i+1][1] - pts[i][1]
                        )
            ax1.arrow(*plot_pts, head_width=0.02, head_length=0.03, 
                    fc='deepskyblue', ec='dodgerblue')

        # Plot for objective function value
        ax2.set_title('Objective Function Value Over Iterations')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Objective Value')
        ax2.plot(range(n_pts), objective_values, label='Final Value', color='blue')
        ax2.plot(range(n_pts), self.calcError(pts), label='Constraint Violation Value', color='red')
        ax2.legend()

        plt.show()

    def calcError(self, pts):
        """Calculate error based on constraint violations

        :param list pts: Points calculated during optimization

        :return list: Error over iterations
        """
        
        errors = []
        for pt in pts:
            # Calculate violation of constraints
            h1_violation = max(0, -self.hx[0](pt))  # Positive if violated
            h2_violation = max(0, -self.hx[1](pt))  # Positive if violated
            total_violation = h1_violation + h2_violation

            errors.append(total_violation)
        return errors

    def getPoints(self, arr, r, t_stop):
        """Perform log-barrier optimization using a sequential least-squares
        minimizer, constrained by the 'barrier' defined by h(x).

        :param tuple arr: Starting point
        :param float r: Alpha
        :param float t_stop: Break point

        :return list[tuple] pts: Points calculated during optimization
        """

        # initialize the points array with the start location
        pts = [arr]
        # format constraints that create the 'barrier' defined by h(x)
        constraints = [{'type': 'ineq', 'fun': h_func} for h_func in self.hx]


        while r >= t_stop:
            # update guess
            c = lambda x: self.fx(x) - r*np.log(self.hx[0](x)) - r*np.log(self.hx[1](x)) 
            # apply sequential least squares minimizer
            res = opt.minimize(c, arr, method='SLSQP', constraints=constraints)         
            self.i += res.nit
            # update current point and append to points list
            arr = tuple(res.x)
            pts.append(arr)
            # reduce step size by half
            r=r/2

        return pts

    def tune_parameters(self, r_range, t_stop_range, start, num_trials=10):
        """Tune 'r' and 't_stop' parameters for the optimizer

        :param tuple r_range: Range of 'r' values to try (min, max)
        :param tuple t_stop_range: Range of 't_stop' values to try (min, max)
        :param tuple start: Initial starting point for optimization
        :param int num_trials: Number of trials for each parameter set, default=10

        :return float best_r: Best 'r' value found
        :return float best_t_stop: Best 't_stop' value found
        """

        best_r = best_t_stop = ""
        best_value = float('inf')

        for r in np.linspace(*r_range, num_trials):
            for t_stop in np.linspace(*t_stop_range, num_trials):
                current_value = 0
                # Run the optimization for the current set of parameters
                for _ in range(num_trials):
                    optimizer = LogBarrierOpt(self.hx, self.fx, start=start, r=r, t_stop=t_stop)
                    # Get the final point of optimization
                    final_pt = optimizer.result[-1]  
                    # Sum up the objective function value
                    current_value += self.fx(final_pt)  
                
                # Average over trials
                current_value /= num_trials  

                # Update best parameters if current is better
                if current_value < best_value:
                    best_value = current_value
                    best_r = r
                    best_t_stop = t_stop 

        print(f"Best 'r' Value: {best_r:.2f}\nBest 't_stop' Value: {best_t_stop}\n")

        return best_r, best_t_stop
    

if __name__ == '__main__':

    # example function optimization
    start = (0.5, 0.5)

    fx = lambda x: (x[0]-1)**2 + 2*(x[1]-2)**2
    hx = (lambda x: 1 - x[0]**2 - x[1]**2, 
          lambda x: x[0] + x[1])
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.warn("deprecated", RuntimeWarning)

        print("\nFunction to Minimize:\n", inspect.getsource(fx))
        print("Contraints:\n", inspect.getsource(hx[0]))

        opt = LogBarrierOpt(hx, fx, start=start)
        pts = opt.result

        opt.plotContour(pts)
        