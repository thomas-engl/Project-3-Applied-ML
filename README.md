# Project-3-Applied-ML

Members: Lars Bosch, Philipp Brückelt and Thomas Engl

This project contains pytorch code for a physics-informed neural network (PINN) to solve partial differential equations. We analyze its performance on the one- and two-dimensional heat equation and compare it to the finite element method (FEM).

To run the code we used a docker container. We recommend to open the container in Visual Studio Code, therefore the Docker extension and Docker itself need to be installed. Then the used packages get automatically installed. To run the notebooks, the dolfinx-env must be used.

# Structure:

Code/pinn.py: Object oriented implementation of a neural network which can solve the d- dimensional heat equation

Code/pinn\_functions.py: Functions to analyze the role of learning rate, regularization, activation functions and to plot the pinn solution

Code/pinn\_run.ipynb: Notebook to run the functions from Code/pinn\_functions.py

Code/1d\_fem\_heat\_equation.ipynb: computes a FEM solution to the 1D heat equation

Code/2d\_heat\_eq\_fem.ipynb: Computes a FEM solution to the 2D hat equation

Code/layers\_test.ipynb: test different layer combinations for the PINN

Code/test\_plots.ipynb: plot the solution obtained by our PINN for concrete examples

