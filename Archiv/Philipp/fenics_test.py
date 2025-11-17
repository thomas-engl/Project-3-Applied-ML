
import dolfinx
import ufl
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

# 1. Mesh erstellen (Einheitsquadrat 32x32)
mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 32, 32)

# 2. Funktionraum (CG1 = kontinuierlich, linear)
V = dolfinx.fem.FunctionSpace(mesh, ("CG", 1))

# 3. Randbedingung: u = 0 auf dem Rand
u_bc = dolfinx.fem.Function(V)
with u_bc.vector.localForm() as loc:
    loc.set(0.0)

def boundary(x):
    return np.full(x.shape[1], True)

bc = dolfinx.fem.dirichletbc(u_bc, dolfinx.fem.locate_dofs_geometrical(V, boundary))

# 4. Variationsproblem definieren
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Rechte Seite f(x, y) = 1
f = dolfinx.fem.Constant(mesh, 1.0)

a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx

# 5. Lösen
problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "cg"})
uh = problem.solve()

# 6. Ergebnisse inspizieren
print("Lösung an den ersten 10 Knoten:", uh.x.array[:10])
