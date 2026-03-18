import struct
from dataclasses import dataclass

import numpy as np
import slangpy as spy

from . import Stage, Mesh, Material, MaterialParameter

class ShaderTableBuilder:
    def __init__(
        self,
    ):
        super().__init__()

        self.modules: list[spy.Module] = []
        self.callable_entries: list[str] = []
    
    def add_module(self, module: spy.Module):
        self.modules.append(module)

    def add_callable(self, entry_point_name: str) -> int:
        index = len(self.callable_entries)
        self.callable_entries.append(entry_point_name)
        return index

class MeshList:
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
        meshes: list[Mesh],
        shader_table_builder: ShaderTableBuilder
    ):
        super().__init__()
        
        self.device = device
        shader_table_builder.add_module(self.device.load_module("shaders/triangle.slang"))

        self.mesh_descs: list[MeshList.MeshDesc] = []
        vertex_count = 0
        index_count = 0
        for mesh in meshes:
            self.mesh_descs.append(MeshList.MeshDesc(
                vertex_offset=vertex_count,
                vertex_count=mesh.vertex_count,
                index_offset=index_count,
                index_count=mesh.index_count,
            ))
            vertex_count += mesh.vertex_count
            index_count += mesh.index_count
        
        positions = np.concatenate([mesh.positions for mesh in meshes], axis=0)
        normals = np.concatenate([mesh.normals for mesh in meshes], axis=0)
        indices = np.concatenate([mesh.indices for mesh in meshes], axis=0)

        mesh_descs_bytes = np.frombuffer(b"".join(desc.pack() for desc in self.mesh_descs), dtype=np.uint8).flatten()
        
        self.position_buffer = self.device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="position_buffer",
            data=positions,
        )
        self.normal_buffer = self.device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="normal_buffer",
            data=normals,
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

            position_offset_pair = spy.BufferOffsetPair(
                buffer=self.position_buffer,
                offset=mesh_desc.vertex_offset * 12,
            )
            index_offset_pair = spy.BufferOffsetPair(
                buffer=self.index_buffer,
                offset=mesh_desc.index_offset * 4,
            )

            blas_input_triangles = spy.AccelerationStructureBuildInputTriangles({
                "vertex_buffers": [position_offset_pair],
                "vertex_format": spy.Format.rgb32_float,
                "vertex_count": mesh_desc.vertex_count,
                "vertex_stride": 4 * 3,
                "index_buffer": index_offset_pair,
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

    def bind(
        self,
        cursor: spy.ShaderCursor,
    ):
        cursor.positions = self.position_buffer
        cursor.normals = self.normal_buffer
        cursor.indices = self.index_buffer
        cursor.descs = self.mesh_desc_buffer



class MaterialList:
    @dataclass
    class MaterialDesc:
        evaluate_call_index: int
        sample_call_index: int
        backpropagate_call_index: int
        requires_grad: int
        constants_start_index: int
        variables_start_index: int

        def pack(self) -> bytes:
            return struct.pack(
                "6I",
                self.evaluate_call_index,
                self.sample_call_index,
                self.backpropagate_call_index,
                self.requires_grad,
                self.constants_start_index,
                self.variables_start_index,
            )

    @dataclass
    class VariableDesc:
        value: float
        range: tuple[float, float]

    def __init__(
        self,
        device: spy.Device,
        materials: list[Material],
        shader_table_builder: ShaderTableBuilder,
    ):
        self.device = device
        self.materials = materials

        constants: list[float] = []
        variables: list[MaterialList.VariableDesc] = []
        self.material_descs: list[MaterialList.MaterialDesc] = []

        module_source = """
        import util;
        import bsdf;
        import scene;

        """

        existing_shaders = set()

        for material in self.materials:

            material_constants: list[float] = []
            material_variables: list[MaterialList.VariableDesc] = []

            for parameter in material.parameters:
                if parameter.requires_grad:
                    material_variables.append(MaterialList.VariableDesc(parameter.value, parameter.range))
                else:
                    material_constants.append(parameter.value)

            constants_start_index = len(constants) 
            variables_start_index = len(variables)

            constants.extend(material_constants)
            variables.extend(material_variables)

            material_requires_grad = len(material_variables) > 0 

            material_hash = abs(hash(material))

            evaluate_entry_point = f"call_evaluate_{material_hash:016x}"
            sample_entry_point = f"call_sample_{material_hash:016x}"
            backpropagate_entry_point = f"call_backpropagate_{material_hash:016x}"

            if material_hash not in existing_shaders:
                module_source += material.evaluate_shader(evaluate_entry_point)
                module_source += material.sample_shader(sample_entry_point)
                module_source += material.backpropagate_shader(backpropagate_entry_point)

                existing_shaders.add(material_hash)

            evaluate_call_index = shader_table_builder.add_callable(evaluate_entry_point)
            sample_call_index = shader_table_builder.add_callable(sample_entry_point)
            if material_requires_grad:
                backpropagate_call_index = shader_table_builder.add_callable(backpropagate_entry_point)
            else:
                backpropagate_call_index = 0xffffffff

            self.material_descs.append(MaterialList.MaterialDesc(
                evaluate_call_index=evaluate_call_index,
                sample_call_index=sample_call_index,
                backpropagate_call_index=backpropagate_call_index,
                requires_grad=1 if material_requires_grad else 0,
                constants_start_index=constants_start_index,
                variables_start_index=variables_start_index,
            ))

        module_name = f"materials{abs(hash(module_source)):016x}" # hash is a workaround for incorrect caching by slangpy
        shader_table_builder.add_module(self.device.load_module_from_source(module_name, module_source))

        constants_bytes = b"".join(struct.pack("f", c) for c in constants)
        if (len(constants_bytes) == 0): 
            constants_bytes = bytes([0])

        self.constants_buffer = self.device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="material_constants_buffer",
            data=np.frombuffer(constants_bytes, dtype=np.uint8).flatten(),
        )

        variables_bytes = b"".join(struct.pack("f", v.value) for v in variables)
        if (len(variables_bytes) == 0): 
            variables_bytes = bytes([0])
        
        self.variables_buffer = self.device.create_buffer(
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            label="material_variables_buffer",
            data=np.frombuffer(variables_bytes, dtype=np.uint8).flatten(),
        )

        self.gradient_buffer = self.device.create_buffer(
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            label="material_gradient_buffer",
            size=len(variables_bytes),
        )

        material_descs_bytes = b"".join(desc.pack() for desc in self.material_descs)
        if (len(material_descs_bytes) == 0): 
            material_descs_bytes = bytes([0])

        self.material_descs_buffer = device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="material_descs_buffer",
            data=np.frombuffer(material_descs_bytes, dtype=np.uint8).flatten(),
        )

    def bind(
        self,
        cursor: spy.ShaderCursor,
    ):
        cursor.descs = self.material_descs_buffer
        cursor.constants = self.constants_buffer
        cursor.variables = self.variables_buffer
        cursor.gradient = self.gradient_buffer

    def zero_grad(self, command_encoder):
        command_encoder.clear_buffer(self.gradient_buffer)

    def download(self):
        pass

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
        self.stage = stage

        self.meshes = MeshList(self.device, self.stage.meshes, shader_table_builder)
        self.materials = MaterialList(self.device, self.stage.materials, shader_table_builder)

        shader_table_builder.add_module(self.device.load_module("shaders/environment.slang"))

        self.instance_descs = [
            Scene.InstanceDesc(
                mesh_index=instance.mesh_id,
                material_index=instance.material_id,
            ) for instance in stage.instances
        ]

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
        cursor.instance_descs = self.instance_descs_buffer

        self.meshes.bind(cursor.meshes)
        self.materials.bind(cursor.materials)
        self.camera.bind(cursor.camera)

    def zero_grad(self, command_encoder: spy.CommandEncoder):
        self.materials.zero_grad(command_encoder)

    def download(self):
        self.materials.download()
