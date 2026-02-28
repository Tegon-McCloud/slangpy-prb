import pathlib
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import slangpy as spy
import pygltflib
from pygltflib import GLTF2

from . import Transform, PerspectiveCamera, Mesh, Material

class Instance:
    def __init__(
        self,
        mesh_id: int,
        material_id: int,
        transform: Transform,
    ):
        super().__init__()
        self.mesh_id = mesh_id
        self.material_id = material_id
        self.transform = transform

@dataclass 
class GltfMeshDescriptor:
    mesh_handles: list[int]
    material_handles: list[int]

class Stage:
    def __init__(
        self,
        environment: spy.Bitmap | None,
    ):
        super().__init__()
        self.meshes: list[Mesh] = []
        self.materials: list[Material] = []
        self.instances: list[Instance] = []
        self.camera = PerspectiveCamera()
        self.environment = environment

    def add_mesh(self, mesh: Mesh) -> int:
        mesh_id = len(self.meshes)
        self.meshes.append(mesh)
        return mesh_id

    def add_material(self, material: Material) -> int:
        material_id = len(self.materials)
        self.materials.append(material)
        return material_id

    def add_instance(self, instance: Instance) -> int:
        instance_id = len(self.instances)
        self.instances.append(instance)
        return instance_id

    def get_material(self, material_id: int) -> Material:
        return self.materials[material_id]
    
    def replace_material(self, material_id: int, material: Material):
        self.materials[material_id] = material


    def load_gltf(self, path: pathlib.Path):
        gltf = GLTF2().load(path)
        
        gltf_meshes = self._load_gltf_meshes(gltf)

        current_scene = gltf.scenes[gltf.scene]
        
        for node_index in current_scene.nodes:
            self._add_gltf_node(gltf, gltf_meshes, node_index, Transform.identity())
        
    def _add_gltf_node(self, gltf: GLTF2, gltf_meshes: list[GltfMeshDescriptor], node_index: int, parent_transform: Transform):
        node = gltf.nodes[node_index]

        if node.matrix != None:
            local_transform = None
            raise RuntimeError("not implemented")
        else:
            local_transform = Transform.identity()

            if node.translation != None:
                local_transform.translation = spy.float3(node.translation)

            if node.rotation != None:
                local_transform.rotation = spy.quatf(node.rotation)
        
            if node.scale != None:
                local_transform.scale = spy.float3(node.scale)

        transform = parent_transform * local_transform

        if node.mesh != None:
            gltf_mesh = gltf_meshes[node.mesh]
            
            for mesh_handle, material_handle in zip(gltf_mesh.mesh_handles, gltf_mesh.material_handles):
                self.add_instance(Instance(
                    mesh_id=mesh_handle,
                    material_id=material_handle,
                    transform=transform,
                ))


        if node.camera != None:
            camera = gltf.cameras[node.camera]
            
            if camera.perspective != None:
                perspective = camera.perspective
                self.camera = PerspectiveCamera(
                    transform,
                    vfov=perspective.yfov,
                    aspect_ratio=perspective.aspectRatio,
                )

            elif camera.orthographic:
                raise RuntimeError("Not implemented")

    def _load_gltf_meshes(self, gltf: GLTF2) -> list[GltfMeshDescriptor]:

        from . import LambertianMaterial

        gltf_meshes: list[GltfMeshDescriptor] = []

        for gltf_mesh in gltf.meshes:

            material_handle = self.add_material(LambertianMaterial(spy.float3(0.5, 0.5, 0.5)))

            primitive_mesh_handles: list[int] = []
            primitive_material_handles: list[int] = []

            for primitive in gltf_mesh.primitives:
                positions = self._read_gltf_accessor(gltf, primitive.attributes.POSITION)
                normals = self._read_gltf_accessor(gltf, primitive.attributes.NORMAL)

                indices = self._read_gltf_accessor(gltf, primitive.indices)
                indices = indices.astype(np.uint32)
                indices = indices.reshape((-1, 3))

                mesh_handle = self.add_mesh(Mesh(positions, normals, indices))

                primitive_mesh_handles.append(mesh_handle)
                primitive_material_handles.append(material_handle)

            gltf_meshes.append(GltfMeshDescriptor(
                mesh_handles=primitive_mesh_handles,
                material_handles=primitive_material_handles
            ))

        return gltf_meshes
    
    def _read_gltf_accessor(self, gltf: GLTF2, accessor_index: int) -> npt.NDArray:
        accessor = gltf.accessors[accessor_index]
        buffer_view: pygltflib.BufferView = gltf.bufferViews[accessor.bufferView]
        buffer = gltf.buffers[buffer_view.buffer]

        data: bytes = gltf.get_data_from_buffer_uri(buffer.uri)
        view_data = data[buffer_view.byteOffset:buffer_view.byteOffset + buffer_view.byteLength]

        dtype_map: dict[int, type] = {
            pygltflib.BYTE: np.int8,
            pygltflib.UNSIGNED_BYTE: np.uint8,
            pygltflib.SHORT: np.int16,
            pygltflib.UNSIGNED_SHORT: np.uint16,
            pygltflib.UNSIGNED_INT: np.uint32,
            pygltflib.FLOAT: np.float32,
        }

        shape_map: dict[str, list[int]] = {
            pygltflib.SCALAR: [],
            pygltflib.VEC2: [2],
            pygltflib.VEC3: [3],
            pygltflib.VEC4: [4],
            pygltflib.MAT2: [2,2],
            pygltflib.MAT3: [3,3],
            pygltflib.MAT4: [4,4],
        }

        dtype = dtype_map[accessor.componentType]
        shape = [accessor.count] + shape_map[accessor.type]
        
        component_count = np.prod(shape)

        if buffer_view.byteStride == None:
            result = np.frombuffer(view_data, dtype=dtype, count=component_count, offset=accessor.byteOffset)
            result = np.reshape(result, shape=shape)
            return result
        else:
            raise RuntimeError("not implemented")






        





