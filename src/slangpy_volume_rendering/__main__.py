import random
import pathlib

import numpy as np
import slangpy as spy
from PIL import Image
from tqdm import tqdm

from . import Transform, Mesh, Material, Instance, Stage, ShaderTableBuilder, Scene, PerspectiveCamera, LambertianMaterial

class PathTracer:
    def __init__(
        self,
        device: spy.Device,
        shader_table_builder: ShaderTableBuilder,
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

        module_names = [
            path.name
            for path in pathlib.Path("./shaders").iterdir()
            if path.is_file()
        ]

        modules = [device.load_module(name) for name in module_names]

        entry_points = [
            entry_point
            for module in modules
            for entry_point in module.entry_points
            if entry_point.stage in [
                spy.ShaderStage.ray_generation,
                spy.ShaderStage.miss,
                spy.ShaderStage.closest_hit,
                spy.ShaderStage.any_hit,
                spy.ShaderStage.intersection,
                spy.ShaderStage.callable,    
            ]
        ]

        program = device.link_program(modules=modules, entry_points=entry_points)

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
            miss_entry_points=["miss_primary", "miss_occlusion"],
            hit_group_names=["triangle_primary", "triangle_occlusion"],
            callable_entry_points=shader_table_builder.callable_entries
        )



    def execute(
        self,
        command_encoder: spy.CommandEncoder,
        scene: Scene,
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
    quad_id = stage.add_mesh(Mesh.quad())

    material_id = stage.add_material(LambertianMaterial(color=spy.float3(1.0, 0.0, 0.0)))
    # material_id_2 = stage.add_material(MicrofacetMaterial(roughness=0.7))

    instance_transform = Transform.identity()
    instance_transform.rotate_x(np.pi / 2.0)
    instance_transform.rotate_z(np.pi / 4.0)


    stage.add_instance(Instance(quad_id, material_id, instance_transform))
    stage.add_instance(Instance(quad_id, material_id, Transform.identity()))

    shader_table_builder = ShaderTableBuilder()

    scene = Scene(device, stage, shader_table_builder)

    path_tracer = PathTracer(device, shader_table_builder, width=1024, height=1024)
    accumulator = Accumulator(device, width=1024, height=1024)
    
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
