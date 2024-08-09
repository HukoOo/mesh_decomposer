# mesh_decomposer
Mesh convex decomposition using V-HACD and COACD


### (1) Third-party Installation

```
cd third-party/CoACD
pip install coacd
cd ../v-hacd
cd app
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### (2) Requirement Installation
```
pip install trimesh vedo mujoco
```

### (3) Usage
