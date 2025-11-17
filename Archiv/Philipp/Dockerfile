FROM ghcr.io/fenics/dolfinx/dolfinx:v0.9.0

# Arbeitsverzeichnis im Container
WORKDIR /home/fenics/workspace

# Optional: zusätzliche Pakete für Visualisierung oder VS Code
RUN apt-get update && apt-get install -y python3-matplotlib python3-pip && \
    pip3 install --no-cache-dir pyvista matplotlib