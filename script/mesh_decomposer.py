#!/usr/bin/env python
# Utils
import os
import logging
import tempfile
import shutil
import subprocess
from typing import List, Optional, Sequence
from pathlib import Path
from lxml import etree
from termcolor import cprint

# Libs for mesh
import vedo
import trimesh
import mujoco
import coacd

# current file path
PACKAGE_NAME = "mesh_decomposer"
PKG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
VHACD_EXECUTABLE = os.path.join(PKG_PATH, "third-parth/v-hacd/app/build/TestVHACD/")
MESH_PATH = os.path.join(PKG_PATH, "meshes/")
COLLISION_MESH_PATH = os.path.join(MESH_PATH, "collision/")
VISUAL_MESH_PATH = os.path.join(MESH_PATH, "visual/")
VHACD_OUTPUTS = ["decomp.obj", "decomp.stl"]

# 2-space indentation for the generated XML.
XML_INDENTATION = "  "

class MeshConverter():
    def __init__(self, decomposer="COACD") -> None:
        if decomposer == "VHACD":
            self.convex_decomposer = VHACDDecomposer(VHACD_EXECUTABLE)
        elif decomposer == "COACD":
            self.convex_decomposer = COACDDecomposer()
        self.mujoco_writer = None
        self.urdf_writer = None

        self.mesh_format = 'obj'
        self.process_mtl = False

    def obj2stl(self, file, output_path):
        pass

    def stl2obj(self, file_path, output_dir=None):
        if type(file_path) == str or type(file_path) == Path():
            mesh = self.load_mesh(file_path)
        else:
            mesh = file_path

        if output_dir == None:
            file_name, file_format = os.path.splitext(Path(file_path).name)
            output_dir = COLLISION_MESH_PATH + file_name + '/'
            output_path =  output_dir + file_name +'.obj'
        else:
            file_name, file_format = os.path.splitext(Path(file).name)
            output_path = output_dir + file_name +'.obj'

        if not Path(output_dir).exists():
            os.mkdir(output_dir)

        if self.save_mesh(mesh, output_path):
            return output_path
        else:
            return False

    def check_mtl(self, obj):
        with open(obj, "r") as f:
            for line in f.readlines():
                if line.startswith("mtllib"):  # Deals with commented out lines.
                    self.process_mtl = True
                    name = line.split()[1]
                    break

    def load_mesh(self, file):
        file_name, file_format = os.path.splitext(Path(file).name)
        file_root = str(Path(file).parent)
        self.mesh_format = file_format[1:]
        mesh = vedo.load(file)
        return mesh

    def save_mesh(self, mesh, output_path, format='obj'):
        vedo.write(mesh, output_path, binary=True)
        return True

    def decompose_convex(self, file):
        file_dir = file
        file_name, file_format = os.path.splitext(Path(file_dir).name)
        file_root = str(Path(file_dir).parent)

        if file_format != self.mesh_format:
            # convert into obj
            file_dir = self.stl2obj(file_dir)
        else:
            # Copy original mesh into save path
            save_path = Path(COLLISION_MESH_PATH + file_name)
            if not save_path.exists():
                os.mkdir(save_path)
            shutil.copy(file_dir, save_path)

        self.convex_decomposer.decompose_convex(file_dir)

class COACDDecomposer():
    def __init__(self) -> None:
        self.threshold = 0.05
        self.max_convex_hull = -1 # max # convex hulls in the result, -1 for no limit, works only when merge is enabled
        self.preprocess_mode = "auto"
        self.prep_resolution = 50 # Preprocessing resolution.
        self.resolution = 2000
        self.mcts_node = 20 # Number of cut candidates for MCTS.
        self.mcts_iteration = 150 # Number of MCTS iterations.
        self.mcts_max_depth = 3 # Maximum depth for MCTS search
        self.pca = True # Use PCA to align input mesh. Suitable for non-axis-aligned mesh
        self.no_merge = True
        self.seed = 0 # random seed

    def decompose_convex(self, file):
        file_name, file_format = os.path.splitext(Path(file).name)

        # Create save path if not exist
        save_path = Path(COLLISION_MESH_PATH + file_name)
        if not save_path.exists():
            os.mkdir(save_path)

        mesh = trimesh.load(file, force="mesh")
        mesh = coacd.Mesh(mesh.vertices, mesh.faces)
        result = coacd.run_coacd(
            mesh,
            threshold=self.threshold,
            max_convex_hull=self.max_convex_hull,
            preprocess_mode=self.preprocess_mode,
            preprocess_resolution=self.prep_resolution,
            resolution=self.resolution,
            mcts_nodes=self.mcts_node,
            mcts_iterations=self.mcts_iteration,
            mcts_max_depth=self.mcts_max_depth,
            pca=self.pca,
            merge=not self.no_merge,
            seed=self.seed,
        )
        mesh_parts = []
        for vs, fs in result:
            mesh_parts.append(trimesh.Trimesh(vs, fs))

        for i, p in enumerate(mesh_parts):
            tmp_scene = trimesh.Scene()
            tmp_scene.add_geometry(p)
            save_file = str(save_path)  + f"/{file_name}_collision_{i}{file_format}"
            tmp_scene.export(save_file)
        return True

class VHACDDecomposer():
    enable: bool = False
    """enable convex decomposition using V-HACD"""
    max_output_convex_hulls: int = 64
    """maximum number of output convex hulls"""
    voxel_resolution: int = 1_000_000
    """total number of voxels to use"""
    volume_error_percent: float = 1.0
    """volume error allowed as a percentage"""
    max_recursion_depth: int = 14
    """maximum recursion depth"""
    disable_shrink_wrap: bool = False
    """do not shrink wrap output to source mesh"""
    fill_mode: str = 'flood'
    """fill mode ;flood', 'surface' and 'raycast' """
    max_hull_vert_count: int = 64
    """maximum number of vertices in the output convex hull"""
    disable_async: bool = False
    """do not run asynchronously"""
    min_edge_length: int = 2
    """minimum size of a voxel edge"""
    split_hull: bool = False
    """try to find optimal split plane location"""

    def __init__(self, executable) -> None:
        self.VHACD_EXECUTABLE = executable

    def decompose_convex(self, file: str) -> bool:
        file_name, file_format = os.path.splitext(Path(file).name)
        file_root = str(Path(file).parent)

        logging.info(f"Decomposing {file}")

        with tempfile.TemporaryDirectory() as tmpdirname:
            prev_dir = os.getcwd()
            os.chdir(tmpdirname)

            # Copy the obj file to the temporary directory.
            shutil.copy(file, tmpdirname)

            # Call V-HACD, suppressing output.
            ret = subprocess.run(
                [
                    f"{self.VHACD_EXECUTABLE}",
                    tmpdirname +'/'+ Path(file).name,
                    "-o",
                        "obj",
                    "-h",
                        f"{self.max_output_convex_hulls}",
                    "-r",
                        f"{self.voxel_resolution}",
                    "-e",
                        f"{self.volume_error_percent}",
                    "-d",
                        f"{self.max_recursion_depth}",
                    "-s",
                        f"{int(not self.disable_shrink_wrap)}",
                    "-f",
                        f"{self.fill_mode.lower()}",
                    "-v",
                        f"{self.max_hull_vert_count}",
                    "-a",
                        f"{int(not self.disable_async)}",
                    "-l",
                        f"{self.min_edge_length}",
                    "-p",
                        f"{int(self.split_hull)}",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                check=True,
            )
            if ret.returncode != 0:
                logging.error(f"V-HACD failed on {Path(file).name}")
                return False

            # Remove the original obj file and the V-HACD output files in temp folder.
            for name in VHACD_OUTPUTS + [Path(file).name]:
                file_to_delete = Path(tmpdirname) / name
                if file_to_delete.exists():
                    file_to_delete.unlink()

            os.chdir(prev_dir)

            # Get list of sorted collisions from temp folder.
            collisions = list(Path(tmpdirname).glob("*.obj"))
            collisions.sort(key=lambda x: x.stem)

            # Create save path if not exist
            save_path = Path(COLLISION_MESH_PATH + file_name)
            if not save_path.exists():
                os.mkdir(save_path)

            # Copy collisions into save path
            for i, filename in enumerate(collisions):
                save_file = str(save_path)  + f"/{file_name}_collision_{i}{filename.suffix}"
                shutil.move(str(filename), save_file)

            # Copy original mesh into save path
            #shutil.copy(file_path, save_path)

        return True

class URDFConverter():

    def __init__(self) -> None:
        self.collision_exists = True
        self.save_urdf = True

    def obj2urdf(self, work_dir:Path):
        file_name = work_dir.name
        obj_file = work_dir / (file_name + '.obj')

        # create mesh from file
        logging.info("Processing OBJ file with trimesh")
        mesh = trimesh.load(
            obj_file,
            split_object=True,
            group_material=True,
            process=False,
            # Note setting this to False is important. Without it, there are a lot of weird
            # visual artifacts in the texture.
            maintain_order=False,
        )

        # Build URDF.
        root = etree.Element("robot", name=obj_file.stem)
        base_link = obj_file.stem+'_link'
        base_link_elem = etree.SubElement(root, "link", name=base_link)

        kwargs = dict(type="mesh", contype="0", conaffinity="0", group="2")

        # Find collision files from the decomposed convex hulls.
        collisions = []
        if self.collision_exists:
            collisions = [
                x for x in work_dir.glob("**/*") if x.is_file() and "collision" in x.name
            ]
            collisions.sort(key=lambda x: int(x.stem.split("_")[-1]))


        for obj in collisions: 
            mesh_path = 'package://' + PACKAGE_NAME + "/meshes/collision/" + file_name + "/" + obj.name

            # link
            link_elem = etree.SubElement(root, "link", name=obj.stem)
            link_visual_elem = etree.SubElement(link_elem, "visual")
            link_geometry_elem = etree.SubElement(link_visual_elem, "geometry")
            link_mesh_elem = etree.SubElement(link_geometry_elem, "mesh", filename=mesh_path)

            link_collision_elem = etree.SubElement(link_elem, "collision")
            link_geometry_elem = etree.SubElement(link_collision_elem, "geometry")
            link_mesh_elem = etree.SubElement(link_geometry_elem, "mesh", filename=mesh_path)

            # link_material_elem = etree.SubElement(link_geometry_elem, "material", name=obj.stem)
            # link_color_elem = etree.SubElement(link_material_elem, "color", rgba=f"{random.uniform(0, 1)} {random.uniform(0, 1)} {random.uniform(0, 1)} 0.5")

            # joint 
            joint_elem = etree.SubElement(root, "joint", name=obj.stem, type='fixed')
            joint_parent_elem = etree.SubElement(joint_elem, "parent", link=base_link)
            joint_child_elem = etree.SubElement(joint_elem, "child", link=obj.stem)
            joint_origin_elem = etree.SubElement(joint_elem, "origin", xyz="0 0 0")


        tree = etree.ElementTree(root)
        etree.indent(tree, space=XML_INDENTATION, level=0)

        # Write urdf file.
        if self.save_urdf:
            xml_path = str(work_dir / f"{obj_file.stem}.urdf")
            tree.write(xml_path, encoding="utf-8")
            logging.info(f"Saved URDF to {xml_path}")

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    mc = MeshConverter()
    urdf = URDFConverter()

    # Find collision files from the decomposed convex hulls.
    mesh_dir = Path(VISUAL_MESH_PATH)
    meshes = [x for x in mesh_dir.glob("**/*") if x.is_file() and (".obj" in x.name or ".stl" in x.name)]
    #meshes.sort(key=lambda x: int(x.stem.split("_")[-1]))
    
    for i in meshes:
        file = str(i)
        # separate file name
        file_name, file_format = os.path.splitext(Path(file).name)
        file_root = str(Path(file).parent)

        mc.decompose_convex(file)

        # create URDF and MJCF files
        work_dir = COLLISION_MESH_PATH + file_name
        urdf.obj2urdf(Path(work_dir))

