import numpy as np
import numpy.typing as npt
import slangpy as spy
from PIL import Image
from tqdm import tqdm

import random

import pathlib
from dataclasses import dataclass

class Transform:
    def __init__(self):
        super().__init__()

        self.translation = spy.float3(0)
        self.rotation = spy.quatf.identity()
        self.scale = spy.float3(1)

    @staticmethod
    def identity() -> 'Transform':
        return Transform()

    @staticmethod
    def from_xyz(x: float, y: float, z: float) -> 'Transform':
        transform = Transform()
        transform.translation = spy.float3(x, y, z)
        
        return transform

    def rotate_x(self, angle: float):
        half_angle = 0.5 * angle
        rotation = spy.quatf(spy.math.sin(half_angle), 0.0, 0.0, spy.math.cos(half_angle))
        self.rotation = spy.math.mul(rotation, self.rotation)

    def rotate_y(self, angle: float):
        half_angle = 0.5 * angle
        rotation = spy.quatf(0.0, spy.math.sin(half_angle), 0.0, spy.math.cos(half_angle))
        self.rotation = spy.math.mul(rotation, self.rotation)
    
    def rotate_z(self, angle: float):
        half_angle = 0.5 * angle
        rotation = spy.quatf(0.0, 0.0, spy.math.sin(half_angle), spy.math.cos(half_angle))
        self.rotation = spy.math.mul(rotation, self.rotation)

    def translate(self, vector: spy.float3):
        self.translation = self.translation + vector

    def to_matrix(self) -> spy.float3x4:
        rotation_matrix = spy.math.matrix_from_quat(self.rotation)

        matrix = spy.float3x4()
        matrix.set_col(0, rotation_matrix.get_col(0) * self.scale.x)
        matrix.set_col(1, rotation_matrix.get_col(1) * self.scale.y)
        matrix.set_col(2, rotation_matrix.get_col(2) * self.scale.z)
        matrix.set_col(3, self.translation)

        return matrix

    def transform_vector(self, vector: spy.float3) -> spy.float3:
        vector = self.scale * vector
        vector = self.rotation * vector
        return vector

    def transform_point(self, point: spy.float3) -> spy.float3:
        point = self.scale * point
        point = self.rotation * point
        point += self.translation
        return point

class PerspectiveCamera:
    def __init__(
        self,
        transform: Transform,
        vfov: float,
        aspect_ratio: float,
    ):
        super().__init__()
        self.transform = transform
        self.vfov = vfov
        self.aspect_ratio = aspect_ratio

    def bind(self, cursor: spy.ShaderCursor):
        matrix = self.transform.to_matrix()
        y_scale = spy.math.tan(0.5 * self.vfov)
        x_scale = self.aspect_ratio * y_scale

        cursor.horizontal = matrix.get_col(0) * x_scale
        cursor.vertical = matrix.get_col(1) * y_scale
        cursor.forward = -matrix.get_col(2)
        cursor.position = matrix.get_col(3)

class Mesh:
    def __init__(
        self,
        vertices: npt.NDArray[np.float32],
        indices: npt.NDArray[np.uint32],
    ):
        super().__init__()
        self.vertices = vertices
        self.indices = indices

    @property
    def vertex_count(self):
        return self.vertices.shape[0]

    @property
    def triangle_count(self):
        return self.indices.shape[0]

    @property
    def index_count(self):
        return self.triangle_count * 3

    @staticmethod
    def quad():
        vertices = np.array([
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ], dtype=np.float32)

        indices = np.array([
            [0, 1, 2],
            [3, 2, 1],
        ], dtype=np.uint32)

        return Mesh(vertices=vertices, indices=indices)

class Instance:
    def __init__(
        self,
        mesh_id: int,
        transform: Transform,
    ):
        super().__init__()
        self.mesh_id = mesh_id
        self.transform = transform

class Stage:
    def __init__(
        self,
        camera: PerspectiveCamera,
        environment: spy.Bitmap | None,
    ):
        super().__init__()
        self.meshes: list[Mesh] = []
        self.instances: list[Instance] = []
        self.camera = camera
        self.environment = environment

    def add_mesh(self, mesh: Mesh) -> int:
        mesh_id = len(self.meshes)
        self.meshes.append(mesh)
        return mesh_id

    def add_instance(self, instance: Instance) -> int:
        instance_id = len(self.instances)
        self.instances.append(instance)
        return instance_id

class Scene:
    @dataclass
    class MeshDesc:
        vertex_count: int
        index_count: int
        vertex_offset: int
        index_offset: int

    def __init__(
        self,
        device: spy.Device,
        stage: Stage,
    ):
        super().__init__()
        self.device = device

        mesh_descs: list[Scene.MeshDesc] = []
        vertex_count = 0
        index_count = 0
        for mesh in stage.meshes:
            mesh_descs.append(Scene.MeshDesc(
                vertex_count=mesh.vertex_count,
                index_count=mesh.index_count,
                vertex_offset=vertex_count,
                index_offset=index_count,
            ))
            vertex_count += mesh.vertex_count
            index_count += mesh.index_count
        
        vertices = np.concatenate([mesh.vertices for mesh in stage.meshes], axis=0)
        indices = np.concatenate([mesh.indices for mesh in stage.meshes], axis=0)

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

        self.blases: list[spy.AccelerationStructure] = []

        command_encoder = self.device.create_command_encoder()
        for mesh_desc in mesh_descs:
            blas_input_triangles = spy.AccelerationStructureBuildInputTriangles({
                "vertex_buffers": [self.vertex_buffer],
                "vertex_format": spy.Format.rgb32_float,
                "vertex_count": mesh_desc.vertex_count,
                "vertex_stride": vertices.itemsize * 3,
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

        self.device.submit_command_buffer(command_encoder.finish())

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
                    "acceleration_structure": self.blases[instance.mesh_id].handle,
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
        self.camera.bind(cursor.camera)

# @dataclass
# class HitGroupDesc:
#     any_hit_entry_point: str | None = None,
#     closest_hit_entry_point: str | None = None,

# class ShaderTableDesc:

#     def __init__(self):
#         super().__init__()

#         self.hit_groups: dict[str, HitGroupDesc] = {}

#         self.ray_gen_entries: list[str] = []
#         self.miss_entries: list[str] = []
#         self.hit_group_entries: list[str] = []
#         self.callable_entries: list[str] = []

#     def add_ray_gen(self, entry_point: str) -> int:
#         index = len(self.ray_gen_entries)
#         self.ray_gen_entries.append(entry_point)
#         return index

#     def add_miss(self, entry_point: str) -> int:
#         index = len(self.miss_entries)
#         self.miss_entries.append(entry_point)
#         return index
    
#     def add_hit_group(
#             self,
#             hit_group_name: str,
#             hit_group: HitGroupDesc,
#         ) -> int:

#         self.hit_groups[hit_group_name] = hit_group 
        
#         index = len(self.hit_group_entries)
#         self.hit_group_entries.append(hit_group_name)
#         return index

#     def add_callable(self, entry_point: str) -> int:
#         index = len(self.callable_entries)
#         self.callable_entries.append(entry_point)
#         return index

#     def hit_group_descs(self) -> list[spy.HitGroupDesc]:


#     def all_entry_points(self) -> list[str]:
#         entry_points: set[str] = set()

#         entry_points.union(self.ray_gen_entries)
#         entry_points.union(self.miss_entries)
#         entry_points.union(self.callable_entries)

#         for hit_group in self.hit_groups:
            
        





class PathTracer:
    def __init__(
        self,
        device: spy.Device,
        width: int,
        height: int,
    ):
        super().__init__()

        self.device = device

        self.sample_texture: spy.Texture = device.create_texture(
            format=spy.Format.rgba32_float,
            width=width,
            height=height,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
            label="sample_texture",
        )

        program = device.load_program("path_trace.slang", ["ray_gen", "miss", "closest_hit_triangle_primary", "any_hit_triangle_occlusion"])

        self.pipeline = device.create_ray_tracing_pipeline(
            program=program,
            hit_groups=[
                spy.HitGroupDesc(hit_group_name="triangle_primary", closest_hit_entry_point="closest_hit_triangle_primary"),
                spy.HitGroupDesc(hit_group_name="triangle_occlusion", any_hit_entry_point="any_hit_triangle_occlusion")
            ],
            max_recursion=2,
            max_ray_payload_size=64,
        )
        self.shader_table = device.create_shader_table(
            program=program,
            ray_gen_entry_points=["ray_gen"],
            miss_entry_points=["miss", "miss_occlusion"],
            hit_group_names=["triangle_primary", "triangle_occlusion"],
            callable_entry_points=[]
        )

    def execute(
        self,
        command_encoder: spy.CommandEncoder,
        scene: Scene
    ):
        with command_encoder.begin_ray_tracing_pass() as pass_encoder:
            shader_object = pass_encoder.bind_pipeline(self.pipeline, self.shader_table)
            cursor = spy.ShaderCursor(shader_object)
            cursor.sample_seed = random.getrandbits(32)
            cursor.sample_texture = self.sample_texture
            scene.bind(cursor.scene)

            pass_encoder.dispatch_rays(0, [1024, 1024, 1])


class Accumulator:

    def __init__(
        self,
        device: spy.Device,
        width: int,
        height: int,
    ):
        super().__init__()
        self.device = device
        self.program = self.device.load_program("accumulate.slang", ["main"])
        self.kernel = self.device.create_compute_kernel(self.program)
        self.accumulation = self.device.create_texture(
            format=spy.Format.rgba32_float,
            width=width,
            height=height,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
            label="accumulation",
        )

    def execute(
        self,
        command_encoder: spy.CommandEncoder,
        sample: spy.Texture,
    ):
        self.kernel.dispatch(
            thread_count=[self.accumulation.width, self.accumulation.height, 1],
            vars={
                "accumulator": {
                    "sample": sample,
                    "accumulation": self.accumulation,
                },
            },
            command_encoder=command_encoder,
        )


def main():
    device = spy.create_device(
        include_paths=[pathlib.Path("./shaders")],
        enable_debug_layers=True,
        type=spy.DeviceType.vulkan
    )

    camera_transform = Transform.from_xyz(0.0, 1.0, 3.0)

    stage = Stage(
        camera=PerspectiveCamera(camera_transform, np.pi / 2.0, 1.0),
        environment=spy.Bitmap.load_from_file("./assets/kloppenheim_06_puresky_4k.hdr"),
    )
    mesh_id = stage.add_mesh(Mesh.quad())

    instance_transform = Transform.identity()
    instance_transform.rotate_x(np.pi / 2.0)
    
    stage.add_instance(Instance(mesh_id, instance_transform))
        
    scene = Scene(device, stage)

    path_tracer = PathTracer(device, 1024, 1024)
    accumulator = Accumulator(device, 1024, 1024)
    
    random.seed(1234)

    for _ in tqdm(range(128)):
        command_encoder = device.create_command_encoder()

        path_tracer.execute(command_encoder, scene)
        accumulator.execute(command_encoder, path_tracer.sample_texture)
        
        device.submit_command_buffer(command_encoder.finish())
        

    result = accumulator.accumulation.to_numpy()
    result = result / result[:,:,3:4]
    result = np.clip(result, 0.0, 1.0)
    result = (result * 255).astype(np.uint8)

    Image.fromarray(result).save("./output/example.png")     

if __name__ == "__main__":
    main()
