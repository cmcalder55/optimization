
import inspect

import numpy as np

import matplotlib.pyplot as plt


class NewtonsDescent():
    """Newton's gradient descent method; steepest descent
    """
    
    def __init__(self, k=10, epsilon=1e-5, t_stop=1e-6, alpha=1e-8, print_out=True, plot_out=True):
        self.printout = print_out
        self.plotout = plot_out
        self.k = k
        self.epsilon = epsilon
        self.t_stop = t_stop
        self.alpha = alpha

    def steepest_descent(self, minimizer):
        """"""

        dim = len(inspect.getfullargspec(minimizer).args)
        start = np.ones(dim)

        costfxn = {}

        x = start
        fx_current = minimizer(*x)

        for i in range(self.k):
            # get the gradient function
            gk = self.gradient(minimizer, x)
            
            # get the Hessian matrix
            H = self.hessian(minimizer, x, dim)

            # calculate step scaling factor 
            alpha = self.get_alpha(gk, H)
            # calculate step size in the gradient direction
            delta = -alpha * gk

            if np.linalg.norm(delta) < self.t_stop:
                # if step size is within t, stop
                break

            # take a step
            x += delta
            # update point 
            fx_current = minimizer(*x)
            costfxn[i] = fx_current

        # print results
        if self.printout:
            self._print_output(x, minimizer, H)
        # plot results
        if self.plotout:
            self._plot_iter(costfxn)

        return costfxn

    def gradient(self, minimizer, x):
        """"""

        grad = np.zeros_like(x)

        x_epsilons = x[:, np.newaxis] + self.epsilon * np.eye(len(x)).T

        grad = (minimizer(*x_epsilons) - minimizer(*x)) / self.epsilon

        return grad

    def hessian(self, minimizer, x, n):
        """"""
        
        H = np.zeros((n, n))
        fx = minimizer(*x)

        # Compute diagonal elements
        for i in range(n):
            x_plus = np.array(x)
            x_minus = np.array(x)
            x_plus[i] += self.epsilon
            x_minus[i] -= self.epsilon
            f_plus = minimizer(*x_plus)
            f_minus = minimizer(*x_minus)
            H[i, i] = (f_plus - 2 * fx + f_minus) / self.epsilon**2

        # Compute off-diagonal elements
        for i in range(n):
            for j in range(i + 1, n):
                # Perturbations are now being made on the same copied array for efficiency
                x_ij = np.array(x)
                
                x_ij[i] += self.epsilon
                x_ij[j] += self.epsilon
                f_pp = minimizer(*x_ij)

                x_ij[j] -= 2 * self.epsilon
                f_pm = minimizer(*x_ij)

                x_ij[i] -= 2 * self.epsilon
                f_mm = minimizer(*x_ij)

                x_ij[j] += 2 * self.epsilon
                f_mp = minimizer(*x_ij)

                H[i, j] = H[j, i] = (f_pp - f_pm - f_mp + f_mm) / (4 * self.epsilon**2)

        return H

    def get_alpha(self, gk, H):
        """Get alpha (smoothing) adjustment
        """
        
        # Use in-place operations to reduce memory footprint
        gk_H_gk = np.dot(gk, H @ gk)
        
        alpha = np.dot(gk, gk) / (gk_H_gk + self.alpha)
        
        return alpha

    def _plot_iter(self, costfxn):
        iterations = list(costfxn.keys())
        objective_values = list(costfxn.values())

        plt.figure()  
        plt.plot(iterations, objective_values, marker='o', linestyle='-')
        plt.title('Minimization with Steepest Gradient Descent')
        plt.xlabel('Iterations')
        plt.ylabel('f(x)')
        plt.grid(True)

        plt.show()       

    def _print_output(self, x_final, minimizer, H_final):
        print('\nMinimizing x vector: \n' + str(x_final))
        print('\nObjective function value: ' + str(minimizer(*x_final)))
        print('\nFinal Hessian matrix: \n' + str(H_final))

if __name__ == '__main__':

    plt.close("all")  # close existing figures      

    minimizer = lambda x0, x1, x2: (x0 + 5)**2 + (x1 + 8)**2 + (x2 + 7)**2 + 2 * x0**2 * (x1**2 + 2 * x2**2)

    gd = NewtonsDescent()
    gd.steepest_descent(minimizer)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    