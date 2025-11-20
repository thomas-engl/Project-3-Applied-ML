
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
### solve the two-dimensional heat equation in \Omega = (0, 1)^2
### u_t - \kappa \Delta u = f    in \Omega
###                     u = 0    on \partial \Omega \times (0, T)
###               u(0, -) = u_0


### =======================================================================
### create a triangular mesh on the unit square (0, 1)^2 with 128 cells 
### in each direction
### We use a high number of cells since there are large oscillations in the
### initial condition (see later)

msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0), (1.0, 1.0)),
    n=(128, 128),
    cell_type=mesh.CellType.triangle,
)

# use Lagrange elements of degree 1
V = fem.functionspace(msh, ("Lagrange", 1))

# implement homogeneous Dirichlet boundary conditions
# u = 0 on the boundary, i.e., for x \in \{0, 1\} and y \in \{0, 1\}

facets = mesh.locate_entities_boundary(
    msh,
    dim=(msh.topology.dim - 1),
    marker=lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) |
                     np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0),
)

# find degrees of freedom

dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)

# create a fem class representing the boundary conditions

bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V)

# Iterate over all time steps: Choose h = 0.001 as step size
# this ensures that the development of the solution in time can be
# observed well in the animation

h = 0.001

x = ufl.SpatialCoordinate(msh)

### =====================================================================
### initial condition u_0
### u_0 (x, y) = sin(pi x) sin(pi y) + sin(2 pi x) sin(4 pi y)
### create a ufl expression and transform it into a fem function later

u0 = ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1]) + ufl.sin(2*np.pi * x[0]
            ) * ufl.sin(4*np.pi * x[1])

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


### ===============================================================================
### Create an animation using pyvista (https://docs.pyvista.org/).

# Extract dolfinx mesh

cells, cell_types, points = plot.vtk_mesh(V)

# Make the grid

grid = pv.UnstructuredGrid(cells, cell_types, points)

# Add initial scalars

values = u_n.x.array
grid.point_data["values"] = values
base_points = grid.points.copy()

# Plotter

plotter = pv.Plotter(notebook=False, off_screen=True)
actor = plotter.add_mesh(
    grid,
    scalars="values",
    lighting=False,
    show_edges=True,
    scalar_bar_args={"title": "Height"},
    clim=[np.min(lst_solutions[0].x.array), np.max(lst_solutions[0].x.array)],
)

# set camera position and zoom such that plot is well visible

plotter.camera.focal_point = (0, 0, 0.2)
plotter.camera.position = (3, 3, 2)
plotter.camera.zoom(0.9)
plotter.open_movie("2d_heat_equation.mp4")

# Keep original points array

pts = grid.points.copy()

# number of frames should be equal to number of time steps

for frame in range(num_steps):

    points = base_points.copy()
    Z = lst_solutions[frame].x.array

    # update coordinates

    pts[:, 2] = Z.ravel()           # modify Z and 
    grid.points = pts               # update mesh points

    # update scalars

    grid.point_data["values"] = Z
    plotter.write_frame()           # triggers render

plotter.close()

