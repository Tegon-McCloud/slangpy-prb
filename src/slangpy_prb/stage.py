import pathlib
import io
from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt
import slangpy as spy
import imageio.v3 as iio
import pygltflib
from pygltflib import GLTF2

from . import VariableId, TextureId, MeshId, MaterialId, InstanceId, \
    Transform, PerspectiveCamera, Mesh, Texture, Material

@dataclass
class Variable:
    value: float
    range: tuple[float, float]

@dataclass
class Instance:
    mesh_id: MeshId
    material_id: MaterialId
    transform: Transform
    
@dataclass 
class GltfMeshDescriptor:
    mesh_ids: list[MeshId]
    material_ids: list[MaterialId]

@dataclass
class GltfTextureDescriptor:
    texture_id: TextureId

class Stage:
    def __init__(self):
        super().__init__()
        self.variables: list[Variable] = []
        self.textures: list[Texture] = []
        self.meshes: list[Mesh] = []
        self.materials: list[Material] = []
        self.instances: list[Instance] = []
        self.camera = PerspectiveCamera()
        self.environment: TextureId | None = None

    def add_variable(self, variable: Variable) -> VariableId:
        variable_id = VariableId(len(self.variables))
        self.variables.append(variable)
        return variable_id

    def add_texture(self, texture: Texture) -> TextureId:
        texture_id = TextureId(len(self.textures))
        self.textures.append(texture)
        return texture_id

    def add_mesh(self, mesh: Mesh) -> MeshId:
        mesh_id = MeshId(len(self.meshes))
        self.meshes.append(mesh)
        return mesh_id

    def add_material(self, material: Material) -> MaterialId:
        material_id = MaterialId(len(self.materials))
        self.materials.append(material)
        return material_id
    
    def get_material(self, material_id: MaterialId) -> Material:
        return self.materials[material_id.index]
    
    def replace_material(self, material_id: MaterialId, material: Material):
        self.materials[material_id.index] = material

    def add_instance(self, instance: Instance) -> InstanceId:
        instance_id = InstanceId(len(self.instances))
        self.instances.append(instance)
        return instance_id



    def set_environment(self, texture_id: TextureId):
        self.environment = texture_id
    
    def load_gltf(self, path: str | pathlib.Path):
        gltf = GLTF2().load(path)
        
        if gltf is None:
            return

        buffer_data = self._load_gltf_buffers(gltf)
        gltf_textures = self._load_gltf_textures(gltf, buffer_data)
        gltf_meshes = self._load_gltf_meshes(gltf, buffer_data)

        current_scene = gltf.scenes[gltf.scene]
        
        if current_scene.nodes != None:
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
            
            for mesh_id, material_id in zip(gltf_mesh.mesh_ids, gltf_mesh.material_ids):
                self.add_instance(Instance(
                    mesh_id=mesh_id,
                    material_id=material_id,
                    transform=transform,
                ))

        if node.camera != None:
            camera = gltf.cameras[node.camera]
            
            if camera.perspective != None:
                perspective = camera.perspective

                aspect_ratio = 1.0
                if perspective.aspectRatio != None:
                    aspect_ratio = perspective.aspectRatio

                self.camera = PerspectiveCamera(
                    transform,
                    vfov=perspective.yfov,
                    aspect_ratio=aspect_ratio,
                )

            elif camera.orthographic:
                raise RuntimeError("Not implemented")
            
        if node.children != None:
            for child_index in node.children:
                self._add_gltf_node(gltf, gltf_meshes, child_index, transform)

    def _load_gltf_textures(self, gltf: GLTF2, buffer_data: list[bytes]) -> list[GltfTextureDescriptor]:

        texture_ids: list[TextureId] = []

        mime_to_extension = {
            "image/png": ".png",
            "image/jpeg": ".jpeg",
        }

        for gltf_image in gltf.images:
            if gltf_image.bufferView != None:
                view_data = self._get_gltf_buffer_view_data(gltf, buffer_data, gltf_image.bufferView)
                extension = mime_to_extension[gltf_image.mimeType]
               
                image = iio.imread(io.BytesIO(view_data), extension=extension)
                texture_id = self.add_texture(Texture(image))
                texture_ids.append(texture_id)
            else:
                raise RuntimeError("Not implemented")

        gltf_textures: list[GltfTextureDescriptor] = []

        for gltf_texture in gltf.textures:
            if gltf_texture.source == None:
                raise RuntimeError("Invalid GLTF: Texture does not specify source image")

            gltf_textures.append(GltfTextureDescriptor(
                texture_id=texture_ids[gltf_texture.source],
            ))
        
        return gltf_textures

    def _load_gltf_meshes(self, gltf: GLTF2, buffer_data: list[bytes]) -> list[GltfMeshDescriptor]:

        from . import Material

        gltf_meshes: list[GltfMeshDescriptor] = []

        for gltf_mesh in gltf.meshes:

            material_id = self.add_material(Material.lambertian(
                reflectance_r=0.5,
                reflectance_b=0.5,
                reflectance_g=0.5,
            ))

            primitive_mesh_ids: list[MeshId] = []
            primitive_material_ids: list[MaterialId] = []

            for primitive in gltf_mesh.primitives:
                positions = self._read_gltf_accessor(gltf, buffer_data, cast(int, primitive.attributes.POSITION))
                normals = self._read_gltf_accessor(gltf, buffer_data, cast(int, primitive.attributes.NORMAL))

                if primitive.indices == None:
                    raise RuntimeError("Not implemented")

                indices = self._read_gltf_accessor(gltf, buffer_data, primitive.indices)
                indices = indices.astype(np.uint32)
                indices = indices.reshape((-1, 3))

                mesh_id = self.add_mesh(Mesh(positions, normals, indices))

                primitive_mesh_ids.append(mesh_id)
                primitive_material_ids.append(material_id)

            gltf_meshes.append(GltfMeshDescriptor(
                mesh_ids=primitive_mesh_ids,
                material_ids=primitive_material_ids
            ))

        return gltf_meshes
    
    def _read_gltf_accessor(self, gltf: GLTF2, buffer_data: list[bytes], accessor_index: int) -> npt.NDArray:
        accessor = gltf.accessors[accessor_index]
        if accessor.bufferView == None:
            raise RuntimeError("Not implemented")

        buffer_view: pygltflib.BufferView = gltf.bufferViews[accessor.bufferView]
        data = buffer_data[buffer_view.buffer]

        length = buffer_view.byteLength
        offset = buffer_view.byteOffset if buffer_view.byteOffset != None else 0

        view_data = data[offset:offset + length]

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
        accessor_offset = accessor.byteOffset if accessor.byteOffset != None else 0

        if buffer_view.byteStride == None:
            result = np.frombuffer(view_data, dtype=dtype, count=component_count, offset=accessor_offset, like=None)
            result = np.reshape(result, shape=shape)
            return result
        else:
            raise RuntimeError("not implemented")

    def _get_gltf_buffer_view_data(self, gltf: GLTF2, buffer_data: list[bytes], buffer_view_index: int) -> bytes:
        buffer_view = gltf.bufferViews[buffer_view_index]
        data = buffer_data[buffer_view.buffer]
        length = buffer_view.byteLength
        offset = buffer_view.byteOffset if buffer_view.byteOffset != None else 0

        return data[offset:offset + length]

    def _load_gltf_buffers(self, gltf: GLTF2) -> list[bytes]:
        buffer_data: list[bytes] = []

        for buffer in gltf.buffers:
            buffer_data.append(cast(bytes, gltf.get_data_from_buffer_uri(buffer.uri)))

        return buffer_data







        





