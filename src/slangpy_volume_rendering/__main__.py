import random
import pathlib

import numpy as np
import numpy.typing as npt
import slangpy as spy
from PIL import Image
from tqdm import tqdm

from . import Transform, Mesh, Material, Instance, Stage, ShaderTableBuilder, Scene, PerspectiveCamera, LambertianMaterial

class PathTracer:
    def __init__(
        self,
        device: spy.Device,
        shader_table_builder: ShaderTableBuilder,
    ):
        super().__init__()

        self.device = device

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


    def sample(
        self,
        command_encoder: spy.CommandEncoder,
        scene: Scene,
        render_target: Scene,
        sample_index: int,
    ):

        with command_encoder.begin_ray_tracing_pass() as pass_encoder:
            shader_object = pass_encoder.bind_pipeline(self.pipeline, self.shader_table)
            cursor = spy.ShaderCursor(shader_object)

            cursor.sample_seed = random.getrandbits(32)
            cursor.sample_index = sample_index
            cursor.render_target = render_target
            scene.bind(cursor.scene)

            pass_encoder.dispatch_rays(0, [render_target.width, render_target.height, 1])

class Tonemapper:

    def __init__(
        self,
        device: spy.Device,
    ):
        self.device = device
        self.program = self.device.load_program("tonemap.slang", ["main"])

        self.kernel = self.device.create_compute_kernel(self.program)

    def tonemap(
        self,
        command_encoder: spy.CommandEncoder,
        input: spy.Texture,
        output: spy.Texture,    
    ):
        self.kernel.dispatch(
            thread_count=[output.width, output.height, 1],
            vars={
                "tonemapper": {
                    "input": input,
                    "output": output,
                },
            },
            command_encoder=command_encoder,
        )



def render(device: spy.Device, stage: Stage, width: int, height: int, sample_count: int, seed: int) -> spy.Texture:

    shader_table_builder = ShaderTableBuilder()

    scene = Scene(device, stage, shader_table_builder)

    path_tracer = PathTracer(device, shader_table_builder)
    random.seed(seed)

    render_target = device.create_texture(
        format=spy.Format.rgba32_float,
        width=width,
        height=height,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
        label="render_target",
    )

    for sample_index in tqdm(range(sample_count)):
        command_encoder = device.create_command_encoder()

        path_tracer.sample(
            command_encoder,
            scene,
            render_target,
            sample_index,
        )
        
        submit_id = device.submit_command_buffer(command_encoder.finish())
        device.wait_for_submit(submit_id)

    return render_target

def save_img(img: npt.NDArray, filename: str):
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

    camera_transform = Transform.from_xyz(-0.4, 2.0, 4.5)
    camera_transform.look_at(spy.float3(-0.2, 0.5, 0.0), spy.float3(0.0, 1.0, 0.0))
    
    stage = Stage(
        camera=PerspectiveCamera(camera_transform, np.pi / 4.0, 1.0),
        environment=spy.Bitmap.load_from_file("./assets/kloppenheim_06_puresky_4k.hdr"),
    )

    stage.load_gltf("./assets/DragonAttenuation.glb")

    width = 1024
    height = 1024

    reference = render(
        device,
        stage,
        width,
        height,
        sample_count=1 << 10,
        seed=1234,
    )

    tonemapper = Tonemapper(device)
    reference_output = device.create_texture(
        format=spy.Format.rgba32_float,
        width=width,
        height=height,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
        label="reference_output",
    )

    command_encoder = device.create_command_encoder()
    tonemapper.tonemap(command_encoder, reference, reference_output)
    device.submit_command_buffer(command_encoder.finish())

    save_img(reference_output.to_numpy(), filename="./output/reference.png")


if __name__ == "__main__":
    main()
