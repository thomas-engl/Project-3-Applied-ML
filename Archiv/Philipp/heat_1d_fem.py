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


### copied this from the dolfinx tutorial

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

h = 0.001

x = ufl.SpatialCoordinate(msh)

### =====================================================================
### initial condition u_0
### u_0 (x) = sin ( pi * x )
### create a ufl expression and transform it into a fem function later

u0 = ufl.sin(np.pi * x[0]) 

# right hand side
# not necessary, if f = 0, but we want to be able to use other RHS, too

f = fem.Constant(msh, 0.0)

# create a list of solutions
# to add u0, we need to transform it into a dolfin function at first
# set u_n = u_0 (interpolate u0 on the grid to obtain the correct data type)

u_n = fem.Function(V)
u0_ = fem.Expression(u0, V.element.interpolation_points())
u_n.interpolate(u0_)
lst_solutions = [u_n]

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

    problem = LinearProblem(a, l, bcs=[bc], petsc_options={
                            "ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()
    
    # set initial condition of next time step to current solution

    u_n.x.array[:] = uh.x.array

    # add solution at current time step to solutions 

    lst_solutions.append(u_n.copy())


print(lst_solutions[0].x.array)
