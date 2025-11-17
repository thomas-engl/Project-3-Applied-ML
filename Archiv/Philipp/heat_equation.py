
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

# -

# Note that it is important to first `from mpi4py import MPI` to
# ensure that MPI is correctly initialised.

# We create a rectangular {py:class}`Mesh <dolfinx.mesh.Mesh>` using
# {py:func}`create_rectangle <dolfinx.mesh.create_rectangle>`, and
# create a finite element {py:class}`function space
# <dolfinx.fem.FunctionSpace>` $V$ on the mesh.

# +
msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0), (1.0, 1.0)),
    n=(32, 32),
    cell_type=mesh.CellType.triangle,
)
V = fem.functionspace(msh, ("Lagrange", 1))
# -

# The second argument to {py:func}`functionspace
# <dolfinx.fem.functionspace>` is a tuple `(family, degree)`, where
# `family` is the finite element family, and `degree` specifies the
# polynomial degree. In this case `V` is a space of continuous Lagrange
# finite elements of degree 1.
#
# To apply the Dirichlet boundary conditions, we find the mesh facets
# (entities of topological co-dimension 1) that lie on the boundary
# $\Gamma_D$ using {py:func}`locate_entities_boundary
# <dolfinx.mesh.locate_entities_boundary>`. The function is provided
# with a 'marker' function that returns `True` for points `x` on the
# boundary and `False` otherwise.

facets = mesh.locate_entities_boundary(
    msh,
    dim=(msh.topology.dim - 1),
    marker=lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) |
                     np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0),
)

# We now find the degrees-of-freedom that are associated with the
# boundary facets using {py:func}`locate_dofs_topological
# <dolfinx.fem.locate_dofs_topological>`:

dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)

# and use {py:func}`dirichletbc <dolfinx.fem.dirichletbc>` to create a
# {py:class}`DirichletBC <dolfinx.fem.DirichletBC>` class that
# represents the boundary condition:

bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V)

# Iterate over all time steps: Choose h = 0.01 as step size
h = 0.01
# to start, define u0 as initial condition and set it to the solution at the
# current time step afterwards

x = ufl.SpatialCoordinate(msh)
# initial condition
u0 = ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])
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
    # -

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

# +

"""
### --- Animation with matplotlib ---

# --- Create triangulation only once ---
cells, cell_types, points = plot.vtk_mesh(V)
triangles = cells.reshape(-1, 4)[:, 1:4]
tri = Triangulation(points[:, 0], points[:, 1], triangles)

# --- Prepare figure ---
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(8,5))
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_zlabel("$u$")
ax.set_zlim(np.min(lst_solutions[0].x.array), np.max(lst_solutions[0].x.array))

# initial values
values0 = lst_solutions[0].x.array

# --- Create initial surface ---
surf = ax.plot_trisurf(
    tri,
    values0,
    cmap="viridis",
    linewidth=0.2,
    antialiased=True
)

# --- Update function ---
def animate(frame):
    values = lst_solutions[frame].x.array
    # Update Z 
    surf.set_array(values)
    return (surf,)

# --- Run animation ---
anim = animation.FuncAnimation(
    fig, animate, frames=len(lst_solutions),
    interval=50, blit=False
)

HTML(anim.to_jshtml())
"""


"""
try:
    import pyvista

    cells, types, x = plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(cells, types, x)
    grid.point_data["u"] = uh.x.array.real
    grid.set_active_scalars("u")
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True)
    warped = grid.warp_by_scalar()
    plotter.add_mesh(warped)
    if pyvista.OFF_SCREEN:
        pyvista.start_xvfb(wait=0.1)
        plotter.screenshot("uh_poisson.png")
    else:
        plotter.show()
except ModuleNotFoundError:
    print("'pyvista' is required to visualise the solution")
    print("Install 'pyvista' with pip: 'python3 -m pip install pyvista'")
# -
"""

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
    clim=[0, 1],
)

plotter.open_gif("wave.gif")

# Keep original points array
pts = grid.points.copy()

nframe = 100
for frame in range(nframe):
    points = base_points.copy()
    Z = lst_solutions[frame].x.array
    # update coordinates
    pts[:, 2] = Z.ravel()     # modify Z
    grid.points = pts         # update mesh points
    # update scalars
    grid.point_data["values"] = Z

    plotter.write_frame()     # triggers render
    time.sleep(0.1)

plotter.close()
