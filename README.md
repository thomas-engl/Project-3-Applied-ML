# Project-3-Applied-ML

Members: Lars Bosch, Philipp Brückelt and Thomas Engl

This project contains pytorch code for a physics-informed neural network (PINN) to solve partial differential equations. We use  We analyze its performance on the one and two dimensional heat equation and compare it to the finite element method (FEM).

We used a docker container and the used packages can be found in the requirements.txt file and can get installed through 'pip install -r requirements.txt'

# Structure:

Code/pinn.py: Object oriented implementation of a neural network which can solve the d- dimensional heat equation

Code/pinn_functions.py: Functions to analyze the role of learning rate, regularization, activation functions and to plot the pinn solution

Code/pinn_run.ipynb: Notebook to run the functions from Code/pinn_functions.py



