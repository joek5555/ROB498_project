# ROB498_project: Double Pendulum Trajectory Optimization

An inverted double pendulum is an example of a highly nonlinear system that is difficult to control, because one input (force applied to the cart) must drive six states to their desired value (the position and velocity of the cart, first link, and second link). Model Predictive Control is one framework used to design controllers for these highly nonlinear systems. It seeks to find a control action that minimizes a cost function over a control sequence for a finite period of time, or horizon. iLQR, DDP, and MPPI are three varients of MPC control. DDP, or differential Dynamic Programming, is a second-order shooting method that achieves quadratic convergence under certain assumptions. DDP requires both the first and second-order derivatives of the motion model and cost function. The second-order derivatives of the cost function are often the most expensive to compute. iLQR, or Iterative Linear Quadratic Regulator, is very similar to DDP, but only the first-order derivatives of the motion model are used. This reduces the computation time of the algorithm. MPPI, or Model Predictive Path Integral control, is a sampling-based optimizer and does not require any derivatives. Given an initial trajectory (either initialized to be all zeros, random actions, or some guess of the correct trajectory), this trajectory is copied K times, and small perturbations are applied to each action in each of these trajectories. Then the dynamics are rolled out for the K trajectories, and the resulting cost of each trajectory is calculated. Finally, the K trajectories are weighted according to the inverse of their cost, and the weighted average is calculated. This resulting trajectory is returned, and the controller will execute the first action and repeat. 

Using the equations of motion for an inverted double pendulum, a simulator was build that will visualize how the system will behave when a force is applied to the cart. The results of the DDP,. iLQR, and MPPI controllers are shown below. 

The python dependencies for this project are: numpy, matplotlib, torch, celluloid

To run a test, navigate to the directory holding demo.py and execute it. Note that you can specify from the command line what controller you would like to test using the -c flag. By default the DDP controller will be launched.

```python3 demo.py -c DDP```

```python3 demo.py -c iLQR```

```python3 demo.py -c MPPI```  

### DDP Controler

![DDP Controller gif](saved_data/DDP_report.gif)

### iLQR controller

![iLQR Controller gif](saved_data/iLQR_report.gif)

### MPPI controller

![MPPI Controller gif](saved_data/MPPI_report.gif)
