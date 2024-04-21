
import inspect
import warnings

from log_barrier_opt import LogBarrierOpt

if __name__ == '__main__':

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
        