
import importlib.util
import numpy as np

import ufl
from dolfinx import fem, io, mesh, plot
from dolfinx.fem.petsc import LinearProblem
from ufl import ds, dx, grad, inner
import numpy as np
# for plotting
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib import animation, rc, rcParams
from matplotlib.tri import Triangulation
from IPython.display import HTML
from mpi4py import MPI

import time


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
### create mesh on the unit square (0, 1)^2 with 64 cells in each direction

msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0), (1.0, 1.0)),
    n=(128, 128),
    cell_type=mesh.CellType.triangle,
)
# use Lagrange elements of degree 1
V = fem.functionspace(msh, ("Lagrange", 1))

# implement boundary conditions
# u = 0 on the boundary

facets = mesh.locate_entities_boundary(
    msh,
    dim=(msh.topology.dim - 1),
    marker=lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) |
                     np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0),
)

# find degrees of freedom

dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)

# and use {py:func}`dirichletbc <dolfinx.fem.dirichletbc>` to create a
# {py:class}`DirichletBC <dolfinx.fem.DirichletBC>` class that
# represents the boundary condition:

bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V)

# Iterate over all time steps: Choose h = 0.01 as step size
h = 0.001
# to start, define u0 as initial condition and set it to the solution at the
# current time step afterwards

x = ufl.SpatialCoordinate(msh)
# initial condition u_0
# create a ufl expression and transform it into a fem function later
u0 = ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1]) + ufl.sin(2*np.pi * x[0]) * ufl.sin(4*np.pi * x[1])
# right hand side
f = fem.Constant(msh, 0.0)

# create a list of solutions
# to add u0, we need to transform it into a dolfin function at first
u_n = fem.Function(V)
u0_ = fem.Expression(u0, V.element.interpolation_points())
u_n.interpolate(u0_)
lst_solutions = [u_n]

for i in range(100):
    # Next, the variational problem is defined:
    # initialize trial and test function
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)   
    # bilinear form and linear form     
    a = (inner(u, v)) * dx + h * inner(grad(u), grad(v)) * dx
    L = inner(h * f + u_n, v) * dx

    # A {py:class}`LinearProblem <dolfinx.fem.petsc.LinearProblem>` object is
    # created that brings together the variational problem, the Dirichlet
    # boundary condition, and which specifies the linear solver. In this
    # case an LU solver is used. The {py:func}`solve
    # <dolfinx.fem.petsc.LinearProblem.solve>` computes the solution.

    # +
    problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()
    
    # set initial condition of next time step to current solution
    u_n.x.array[:] = uh.x.array

    # add solution at current time step to solutions 
    lst_solutions.append(u_n.copy())


# and displayed using [pyvista](https://docs.pyvista.org/).

# Extract dolfinx mesh
cells, cell_types, points = plot.vtk_mesh(V)

# Make the grid
grid = pv.UnstructuredGrid(cells, cell_types, points)

# Add initial scalars
values = u_n.x.array
grid.point_data["values"] = values
base_points = grid.points.copy()

# Plotter (GIF mode)
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

nframe = 100
for frame in range(nframe):
    points = base_points.copy()
    Z = lst_solutions[frame].x.array
    # update coordinates
    pts[:, 2] = Z.ravel()     # modify Z and 
    grid.points = pts         # update mesh points
    # update scalars
    grid.point_data["values"] = Z
    plotter.write_frame()     # triggers render

plotter.close()

