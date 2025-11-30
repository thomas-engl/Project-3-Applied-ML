import importlib.util
import numpy as np
import ufl
from dolfinx import fem, io, mesh, plot
from dolfinx.fem.petsc import LinearProblem
from ufl import ds, dx, grad, inner
import numpy as np

# for plotting / animation

import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib import animation, rc, rcParams
from matplotlib.tri import Triangulation
from IPython.display import HTML
from mpi4py import MPI

import time


### copied this block from the dolfinx tutorial

if importlib.util.find_spec("petsc4py") is not None:
    import dolfinx

    if not dolfinx.has_petsc:
        print("This demo requires DOLFINx to be compiled with PETSc enabled.")
        exit(0)
    from petsc4py.PETSc import ScalarType  # type: ignore
else:
    print("This demo requires petsc4py.")
    exit(0)


### =======================================================================
### solve the two-dimensional heat equation in \Omega = (0, 1)
### u_t - \kappa \Delta u = f    in \Omega
###     u(0, t) = u(1, t) = 0    for t > 0
###               u(0, -) = u_0


### =======================================================================
### create an equdistant mesh on (0, 1) with 128 cells 

msh = mesh.create_unit_interval(
    comm=MPI.COMM_WORLD,
    nx=128
)

# use Lagrange elements of degree 1
V = fem.functionspace(msh, ("Lagrange", 1))

# implement homogeneous Dirichlet boundary conditions
# u = 0 on the boundary, i.e., for x \in \{0, 1\}

facets = mesh.locate_entities_boundary(
    msh,
    dim=0,
    marker=lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0),
)

# find degrees of freedom

dofs = fem.locate_dofs_topological(V=V, entity_dim=0, entities=facets)

# create a fem class representing the boundary conditions

bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V)

# Iterate over all time steps: Choose h = 0.001 as step size
# this ensures that the development of the solution in time can be
# observed well in the animation

h = 0.005

x = ufl.SpatialCoordinate(msh)

### =====================================================================
### initial condition u_0
### u_0 (x) = sin ( pi * x )
### create a ufl expression and transform it into a fem function later

u0 = ufl.sin(np.pi * x[0]) + ufl.sin(4 * np.pi * x[0])

# right hand side
# not necessary, if f = 0, but we want to be able to use other RHS, too

f = fem.Constant(msh, 0.0)

# create a list of solutions
# to add u0, we need to transform it into a dolfin function at first
# set u_n = u_0 (interpolate u0 on the grid to obtain the correct data type)

u_n = fem.Function(V)
u0_ = fem.Expression(u0, V.element.interpolation_points())
u_n.interpolate(u0_)
lst_solutions = [u_n.copy()]

# diffusivity coefficient \kappa

kappa = 0.1

# number of time steps

num_steps = 200

for i in range(num_steps):

    ### ================================================================
    ### Construct the variational problem
    ### Discretize the time by finite differences to obtain the equation
    ### u^{n+1} = h ( f^{n+1} + \kappa \Delta u^{n+1} ) + u^{n}
    ### where u^{n}, f^{n} are the respective functions at time step n
    ### in the following, u^{n} and f^{n} are simply denoted by u and f,
    ### respectively.
    ### then derive the weak formulation


    # initialize trial and test function

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)  

    ### =================================================================
    ### bilinear form and linear form
    ### a(u, v) = \int_{\Omega} uv + h \kappa \nabla u \cdot \nabla v dx
    ### l(v) = \int_{\Omega} (hf + u^{n})v dx

    # inner denotes the standard inner product, for real-valued functions
    # this is a normal product     

    a = (inner(u, v)) * dx + h * inner(kappa * grad(u), grad(v)) * dx
    l = inner(h * f + u_n, v) * dx

    ### ===============================================================
    ### define the linear problem
    ### a(u, v) = l(v) for all v in H^1_0 ( \Omega )

    # use GMRES (generalised minimal residual) to solve the linear problem
    # incomplete LU decomposition as precoditioner
    problem = LinearProblem(a, l, bcs=[bc], petsc_options={
        "ksp_type": "gmres",
        "ksp_rtol": 1e-8,
        "ksp_max_it": 2000,
        "pc_type": "ilu"})
    uh = problem.solve()
    
    # set initial condition of next time step to current solution

    u_n.x.array[:] = uh.x.array

    # add solution at current time step to solutions 

    lst_solutions.append(u_n.copy())


### ===============================================================================
### analytic solution
### u(t, x) = exp(- k pi^2 t) sin(pi x) + exp(- 16 k pi^2 t) sin(4 pi x)

# create a normal python function since this easier to plot
def u(t, x): 
    return np.exp(- kappa * np.pi**2 * t) * np.sin(np.pi * x) + np.exp(
        - 16 * kappa * np.pi**2 * t) * np.sin(4 * np.pi * x)
# vectorize the function in the spatial coordinate x
u_vec = np.vectorize(u, excluded='t')

### ================================================================================
### compute the L^2 error
### do this for every time step separately

# we have to construct a ufl expression again
# Function space for exact solution
V_exact = fem.functionspace(msh, ("Lagrange", 5))
L_2_errs = []

# leave out 0-th time step since solution is exact by definition
for i in range(1, len(lst_solutions)):
    u_exact = fem.Function(V_exact)
    # t = i * h
    u_e = ufl.exp(- kappa * np.pi**2 * i * h) * ufl.sin(np.pi * x[0]) + ufl.exp(
        - 16 * kappa * np.pi**2 * i * h) * ufl.sin(4 * np.pi * x[0])
    u_exact_expr = fem.Expression(u_e, V_exact.element.interpolation_points())
    u_exact.interpolate(u_exact_expr)

    # L2 errors
    L_2_err = np.sqrt(
        msh.comm.allreduce(
            fem.assemble_scalar(fem.form((lst_solutions[i] - u_exact) ** 2 * ufl.dx)), 
            op=MPI.SUM
            )
        )
    L_2_errs.append(L_2_err)

print("list of L^2 errors: ", L_2_errs)

### ===============================================================================
### Create an animation using matplotlib
### also plot the analytic solution for comparison

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

x = np.linspace(0, 1, 129)
fig, ax = plt.subplots()

# plot FEM and analytic solution together
line_1, = ax.plot(x, lst_solutions[0].x.array, color='blue', label='FE solution')
line_2, = ax.plot(x, u_vec(0, x), '--', color='red', label='analytic solution')
ax.set_ylim(-0.75, 2.0)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$u(t, x)$')
ax.legend(loc='upper right')
# add text which shows the current time step in each frame
time_count = ax.text(0.1, 0.9, 't=0.000', transform=ax.transAxes)

def update(frame):
    if frame <= 10:
        ### keep initial solution for first 10 frames
        y = lst_solutions[0].x.array
        z = u_vec(0, x)
        line_1.set_ydata(y)
        line_2.set_ydata(z)
        time_count.set_text('t=0.000')
    else:
        y = lst_solutions[frame - 10].x.array
        z = u_vec((frame - 10) * h, x)
        line_1.set_ydata(y)
        line_2.set_ydata(z)
        time_count.set_text('t={:.3f}'.format((frame - 10) * h))
    return line_1, line_2, time_count

ani = FuncAnimation(fig, update, frames=len(lst_solutions) + 10, interval=100)
ani.save('1d_heat_equation.gif', writer='pillow')
plt.show()
