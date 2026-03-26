import random

import slangpy as spy
from tqdm import tqdm

from . import Scene, ShaderTableBuilder

class PathTracer:
    def __init__(
        self,
        device: spy.Device,
        shader_table_builder: ShaderTableBuilder,
    ):
        super().__init__()

        self.device = device

        modules = list(shader_table_builder.modules)
        modules.append(self.device.load_module("shaders/path_trace.slang"))

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

    def render(
        self,
        scene: Scene,
        width: int,
        height: int,
        sample_count: int,
        seed: int | None = None,
    ) -> spy.Texture:
        
        if seed != None:
            random.seed(seed)

        render_target = self.device.create_texture(
            format=spy.Format.rgba32_float,
            width=width,
            height=height,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
            label="render_target",
        )

        for sample_index in tqdm(range(sample_count)):
            command_encoder = self.device.create_command_encoder()

            self.sample(
                command_encoder,
                scene,
                render_target,
                sample_index,
            )
            
            submit_id = self.device.submit_command_buffer(command_encoder.finish())
            self.device.wait_for_submit(submit_id)

        return render_target


    def sample(
        self,
        command_encoder: spy.CommandEncoder,
        scene: Scene,
        render_target: spy.Texture,
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