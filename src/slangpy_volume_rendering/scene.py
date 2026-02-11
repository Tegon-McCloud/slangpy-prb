import struct
from dataclasses import dataclass

import numpy as np
import slangpy as spy

from . import Stage, Mesh, Material

class ShaderTableBuilder:
    def __init__(
        self,
    ):
        super().__init__()

        self.callable_entries: list[str] = []

    def add_callable(self, entry_point_name: str) -> int:
        index = len(self.callable_entries)
        self.callable_entries.append(entry_point_name)
        return index


class SceneMeshList:
    @dataclass
    class MeshDesc:
        vertex_offset: int
        vertex_count: int
        index_offset: int
        index_count: int

        def pack(self) -> bytes:
            return struct.pack(
                "II",
                self.vertex_offset,
                self.index_offset,
            )

    def __init__(
        self,
        device: spy.Device,
        meshes: list[Mesh]        
    ):
        super().__init__()
        
        self.device = device

        self.mesh_descs: list[SceneMeshList.MeshDesc] = []
        vertex_count = 0
        index_count = 0
        for mesh in meshes:
            self.mesh_descs.append(SceneMeshList.MeshDesc(
                vertex_offset=vertex_count,
                vertex_count=mesh.vertex_count,
                index_offset=index_count,
                index_count=mesh.index_count,
            ))
            vertex_count += mesh.vertex_count
            index_count += mesh.index_count
        
        vertices = np.concatenate([mesh.vertices for mesh in meshes], axis=0)
        indices = np.concatenate([mesh.indices for mesh in meshes], axis=0)

        mesh_descs_bytes = np.frombuffer(b"".join(desc.pack() for desc in self.mesh_descs), dtype=np.uint8).flatten()
        
        self.vertex_buffer = self.device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="vertex_buffer",
            data=vertices,    
        )
        self.index_buffer = self.device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="index_buffer",
            data=indices,
        )
        self.mesh_desc_buffer = self.device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="mesh_desc_buffer",
            data=mesh_descs_bytes,
        )

        self.blases: list[spy.AccelerationStructure] = []

        command_encoder = self.device.create_command_encoder()
        for mesh_desc in self.mesh_descs:
            blas_input_triangles = spy.AccelerationStructureBuildInputTriangles({
                "vertex_buffers": [self.vertex_buffer],
                "vertex_format": spy.Format.rgb32_float,
                "vertex_count": mesh_desc.vertex_count,
                "vertex_stride": 4 * 3,
                "index_buffer": self.index_buffer,
                "index_format": spy.IndexFormat.uint32,
                "index_count": mesh_desc.index_count,
                "flags": spy.AccelerationStructureGeometryFlags.opaque,
            })

            blas_build_desc = spy.AccelerationStructureBuildDesc({
                "inputs": [blas_input_triangles],
            })

            blas_sizes = self.device.get_acceleration_structure_sizes(blas_build_desc)

            blas_scratch_buffer = self.device.create_buffer(
                size=blas_sizes.scratch_size,
                usage=spy.BufferUsage.unordered_access,
                label="blas_scratch_buffer",
            )

            blas = self.device.create_acceleration_structure(
                size=blas_sizes.acceleration_structure_size,
                label="blas",
            )
            self.blases.append(blas)

            command_encoder.build_acceleration_structure(
                desc=blas_build_desc,
                dst=blas,
                src=None,
                scratch_buffer=blas_scratch_buffer,
            )

        device.submit_command_buffer(command_encoder.finish())



class SceneMaterialList:
    @dataclass
    class MaterialDesc:
        evaluate_call_index: int
        sample_call_index: int
        parameter_address: int

        def pack(self) -> bytes:
            return struct.pack(
                "IIQ",
                self.evaluate_call_index,
                self.sample_call_index,
                self.parameter_address,
            )

    def __init__(
        self,
        device: spy.Device,
        materials: list[Material],
        shader_table_builder: ShaderTableBuilder
    ):
        self.device = device
        
        parameter_bytes = bytearray()
        parameter_offsets: list[int] = []
        parameter_size = 0

        for material in materials:
            bytes = material.pack_parameters()

            parameter_offsets.append(parameter_size)
            parameter_bytes.extend(bytes)
            parameter_size += len(parameter_bytes)

        parameter_bytes = np.frombuffer(parameter_bytes, dtype=np.uint8).flatten()
        
        self.parameter_buffer = self.device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="parameter_buffer",
            data=parameter_bytes,
        )
        
        self.material_descs: list[SceneMaterialList.MaterialDesc] = []

        for material, parameter_offset in zip(materials, parameter_offsets):
            evaluate_call_index = shader_table_builder.add_callable(material.evaluate_entry_point)
            sample_call_index = shader_table_builder.add_callable(material.sample_entry_point)

            self.material_descs.append(SceneMaterialList.MaterialDesc(
                evaluate_call_index=evaluate_call_index,
                sample_call_index=sample_call_index,
                parameter_address=self.parameter_buffer.device_address + parameter_offset,
            ))
        
        material_descs_bytes = np.frombuffer(b"".join(desc.pack() for desc in self.material_descs), dtype=np.uint8).flatten()

        self.material_descs_buffer = device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="material_descs_buffer",
            data=material_descs_bytes,
        )

class Scene:
    @dataclass
    class InstanceDesc:
        mesh_index: int
        material_index: int

        def pack(self) -> bytes:
            return struct.pack(
                "II",
                self.mesh_index,
                self.material_index
            )

    def __init__(
        self,
        device: spy.Device,
        stage: Stage,
        shader_table_builder: ShaderTableBuilder,
    ):
        super().__init__()
        self.device = device

        self.meshes = SceneMeshList(self.device, stage.meshes)
        self.materials = SceneMaterialList(self.device, stage.materials, shader_table_builder)

        self.instance_descs = [Scene.InstanceDesc(
            mesh_index=instance.mesh_id,
            material_index=instance.material_id,
        ) for instance in stage.instances]

        instance_descs_bytes = np.frombuffer(b"".join(desc.pack() for desc in self.instance_descs), dtype=np.uint8).flatten()

        self.instance_descs_buffer = device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="instance_descs_buffer",
            data=instance_descs_bytes,
        )

        instance_list = device.create_acceleration_structure_instance_list(size=len(stage.instances))

        for instance_id, instance in enumerate(stage.instances):
            instance_list.write(
                instance_id,
                {
                    "transform": instance.transform.to_matrix(),
                    "instance_id": instance_id,
                    "instance_mask": 0xff,
                    "instance_contribution_to_hit_group_index": 0,
                    "flags": spy.AccelerationStructureInstanceFlags.none,
                    "acceleration_structure": self.meshes.blases[instance.mesh_id].handle,
                },
            )

        tlas_build_desc = spy.AccelerationStructureBuildDesc({
            "inputs": [instance_list.build_input_instances()]
        })

        tlas_sizes = device.get_acceleration_structure_sizes(tlas_build_desc)

        tlas_scratch_buffer = device.create_buffer(
            size=tlas_sizes.scratch_size,
            usage=spy.BufferUsage.unordered_access,
            label="tlas_scratch_buffer",
        )

        self.tlas = device.create_acceleration_structure(
            size=tlas_sizes.acceleration_structure_size,
            label="tlas",
        )

        command_encoder = device.create_command_encoder()
        command_encoder.build_acceleration_structure(
            desc=tlas_build_desc,
            dst=self.tlas,
            src=None,
            scratch_buffer=tlas_scratch_buffer,
        )
        device.submit_command_buffer(command_encoder.finish())

        texture_loader = spy.TextureLoader(device)
        self.environment_texture = texture_loader.load_texture(stage.environment)

        self.camera = stage.camera


    def bind(self, cursor: spy.ShaderCursor):
        cursor.tlas = self.tlas
        cursor.environment_map = self.environment_texture

        cursor.vertices = self.meshes.vertex_buffer
        cursor.indices = self.meshes.index_buffer
        cursor.mesh_descs = self.meshes.mesh_desc_buffer

        cursor.material_descs = self.materials.material_descs_buffer

        cursor.instance_descs = self.instance_descs_buffer

        self.camera.bind(cursor.camera)