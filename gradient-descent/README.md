# Batch vs. Stochastic Gradient Descent
## Best learning rate
Based on MSE and convergence, the best learning rate for Batch GD is 0.0001 to 0.002 and the best for Stochastic GD is 0.0001 to 0.004. Overall, SGD was able to perform better at higher learning rates than BGD, but BGD sometimes converged faster at very low learning rates.

## Accuracy on the test set

Batch GD was more accurate overall, since it considers every gradient and takes the steepest route. Stochastic GD is only using one point for estimations at each iteration, so it jumps around more in terms of error. This can be seen in the noisiness of the graphs of MSE vs iterations for stochastic GD compared to the smoother curves of batch GD.

## Speed of convergence

SGD converged faster than BGD. This is due to it randomly sampling points to take the gradient, so it can update the weights quickly and can take larger steps towards the optimal point. Since a tolerance was used as a stopping condition, in most cases SGD was able to reach values within an acceptable range faster than BGD. This makes it preferable for large datasets, or when a good estimate of the optimum value is acceptable; the reduction in accuracy can be ok since it converges faster.

# Derive the Gradient
Given $x=\begin{bmatrix}2\\3\end{bmatrix}$
 , and $A = \begin{bmatrix}a_{1} & a_{2}\\a_{3} & a_{4}\end{bmatrix}$

The quadratic equation can be generalized as a series of summations.

$$f(x) = 0.5\sum_{i=1}^n \sum_{j=1}^n x_{i}*A_{ij}*x_{j}$$

Since the lengths of x and A are equal (n = 2), can do the partial derivative w.r.t. matrix element k to compute the gradient.

$$f(x) = 0.5\sum_{i=1}^n (A_{ii}*x_{i}^2 + \sum_{j\neq i} x_{i}*A_{ij}*x_{j})$$
\
$$\nabla f(x) = 0.5\displaystyle \frac{\partial f}{\partial x_{k}}f(x) = 
0.5\binom{\frac{\partial f}{\partial x_{1}}}
      {\frac{\partial f}{\partial x_{2}}} 
= 0.5\binom{\sum_{j=1}^n x_{j}*A_{j1} + \sum_{j=1}^n A_{1j}*x_{j}}
{\sum_{j=1}^n x_{j}*A_{j2} + \sum_{j=1}^n A_{2j}*x_{j}}
= 0.5(A^T+A)x$$
\
If A is symmetric, i.e. $A_{ij} = A_{ji}$ then $A^T = A$ and the gradient becomes:\
$$\nabla f(x) = 0.5(A+A)x = Ax$$
