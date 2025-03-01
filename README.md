# mesh_decomposer
Mesh convex decomposition using V-HACD and COACD

<p float="left">
  <img src="asset/convex_decomposition.png" height="200">
</p>

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

Add concave meshes into meshes/visual and run script:
```
python script/mesh_mesh_decomposer.py
```

Then decomposed meshes are saved in meshes/collision folder with urdf file.

