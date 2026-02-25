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

            pass_encoder.dispatch_rays(0, [self.sample_texture.width, self.sample_texture.height, 1])


class Accumulator:

    def __init__(
        self,
        device: spy.Device,
        width: int,
        height: int,
    ):
        super().__init__()
        self.device = device
        self.accumulate_program = self.device.load_program("accumulate.slang", ["accumulate"])
        self.normalize_program = self.device.load_program("normalize.slang", ["normalize"])

        self.accumulate_kernel = self.device.create_compute_kernel(self.accumulate_program)
        self.normalize_kernel = self.device.create_compute_kernel(self.normalize_program)
        self.accumulation = self.device.create_texture(
            format=spy.Format.rgba32_float,
            width=width,
            height=height,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
            label="accumulation",
        )

    def accumulate(
        self,
        command_encoder: spy.CommandEncoder,
        sample: spy.Texture,
    ):
        self.accumulate_kernel.dispatch(
            thread_count=[self.accumulation.width, self.accumulation.height, 1],
            vars={
                "accumulator": {
                    "sample": sample,
                    "accumulation": self.accumulation,
                },
            },
            command_encoder=command_encoder,
        )

    def normalize(
        self,
        command_encoder: spy.CommandEncoder,
    ):
        self.normalize_kernel.dispatch(
            thread_count=[self.accumulation.width, self.accumulation.height, 1],
            vars={
                "accumulator": {
                    "accumulation": self.accumulation,
                },
            },
            command_encoder=command_encoder,
        )



def render(device: spy.Device, stage: Stage, width: int, height: int, sample_count: int, seed: int) -> spy.Texture:

    shader_table_builder = ShaderTableBuilder()

    scene = Scene(device, stage, shader_table_builder)

    path_tracer = PathTracer(device, shader_table_builder, width, height)
    accumulator = Accumulator(device, width, height)
    
    random.seed(seed)

    for _ in tqdm(range(sample_count)):
        command_encoder = device.create_command_encoder()

        path_tracer.execute(command_encoder, scene)
        accumulator.accumulate(command_encoder, path_tracer.sample_texture)
        
        device.submit_command_buffer(command_encoder.finish())

    command_encoder = device.create_command_encoder()
    accumulator.normalize(command_encoder)
    device.submit_command_buffer(command_encoder.finish())

    return accumulator.accumulation

def save_texture(texture: spy.Texture, filename: str, scale: float = 1.0):
    img = texture.to_numpy()
    img = scale * img
    print(img.min(), img.max())
    img = np.clip(img, 0.0, 1.0)
    img = (img * 255).astype(np.uint8)
    
    Image.fromarray(img).save(filename) 

def main():
    device = spy.create_device(
        include_paths=[pathlib.Path("./shaders")],
        enable_debug_layers=True,
        enable_print=True,
        type=spy.DeviceType.vulkan,
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

    width = 256
    height = 256

    reference = render(
        device,
        stage,
        width,
        height,
        sample_count=1 << 14,
        seed=1234,
    )
    save_texture(reference, filename="./output/reference.png")

    material: LambertianMaterial = stage.get_material(material_id)
    material.color = spy.float3(0.5, 0.5, 0.5)

    primal = render(
        device,
        stage,
        width,
        height,
        sample_count=1 << 14,
        seed=1233,
    )
    save_texture(primal, filename="./output/primal.png")

    reference_arr = reference.to_numpy()
    primal_arr = primal.to_numpy()

    print(reference_arr.max())
    print(primal_arr.max())

    loss = np.mean((primal_arr - reference_arr) * (primal_arr - reference_arr))
    print(f"loss: {loss}")

    adjoint = device.create_texture(
        format=spy.Format.rgba32_float,
        width=width,
        height=height,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
        label="sample_texture",
    )
    adjoint_program = device.load_program("l2_adjoint", ["main"])
    adjoint_kernel = device.create_compute_kernel(adjoint_program)

    command_encoder = device.create_command_encoder()
    adjoint_kernel.dispatch(
        thread_count=[width, height, 1],
        vars={
            "primal": primal,
            "reference": reference,
            "adjoint": adjoint,
        },
        command_encoder=command_encoder,
    )
    device.submit_command_buffer(command_encoder.finish())

    save_texture(adjoint, "./output/adjoint.png", scale=1.0)



if __name__ == "__main__":
    main()
