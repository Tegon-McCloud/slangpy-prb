import slangpy as spy

from . import CodeBuilder

class PostProcessStage:
    def __init__(
        self,
        struct_name: str,
    ):
        self.struct_name = struct_name

    def bind(self, cursor: spy.ShaderCursor): ...

class PostProcessor:
    def __init__(
        self,
        device: spy.Device,
        stages: list[PostProcessStage],
    ):
        self.device = device
        self.stages = stages

        builder = CodeBuilder()

        builder.append_line("import util;")
        builder.append_line("import postprocess;")
        builder.newline()

        for i, stage in enumerate(stages):
            builder.declare(f"ParameterBlock<{stage.struct_name}>", f"stage{i}")
        builder.newline()

        builder.append_line("[Differentiable]")
        builder.append_line("float4 apply_all(float4 color)")
        builder.begin_block()
        for i, stage in enumerate(stages):
            builder.append_line(f"color.rgb = stage{i}.apply(color.rgb);")
        builder.append_line("return color;")
        builder.end_block()
        builder.newline()

        # main entry point
        builder.append_line("[shader(\"compute\")]")
        builder.append_line("[numthreads(16, 16, 1)]")
        builder.append_line("void main(")
        builder.inc_indent()
        builder.append_line("int3 tid: SV_DispatchThreadID,")
        builder.append_line("uniform Texture2D<float4> input,")
        builder.append_line("uniform RWTexture2D<float4> output,")
        builder.dec_indent()
        builder.append_line(")")
        builder.begin_block()
        builder.declare("uint2", "pixel", "tid.xy")
        builder.declare("uint2", "dim", "dimensions(input)")
        builder.append_line("if (pixel.x >= dim.x || pixel.y >= dim.y) return;")
        builder.newline()

        builder.declare("float4", "color", "input[pixel]")
        builder.append_line("color = apply_all(color);")
        builder.append_line("output[pixel] = color;")

        builder.end_block()
        builder.newline()

        # backwards entry point
        builder.append_line("[shader(\"compute\")]")
        builder.append_line("[numthreads(16, 16, 1)]")
        builder.append_line("void backwards(")
        builder.inc_indent()
        builder.append_line("int3 tid: SV_DispatchThreadID,")
        builder.append_line("uniform Texture2D<float4> input,")
        builder.append_line("uniform Texture2D<float4> weight,")
        builder.append_line("uniform RWTexture2D<float4> gradient,")
        builder.dec_indent()
        builder.append_line(")")
        builder.begin_block()
        builder.declare("uint2", "pixel", "tid.xy")
        builder.declare("uint2", "dim", "dimensions(input)")
        builder.append_line("if (pixel.x >= dim.x || pixel.y >= dim.y) return;")
        builder.newline()

        builder.declare("float4", "w", "weight[pixel]")
        builder.declare("DifferentialPair<float4>", "color", "diffPair(input[pixel])")
        builder.append_line("bwd_diff(apply_all)(color, w);")
        builder.append_line("gradient[pixel] = color.d;")

        builder.end_block()
        builder.newline()

        module_source = builder.build()
        module_hash = abs(hash(module_source))

        with open("output/postprocess_module.slang", 'w') as f:
            f.write(module_source)

        self.module = self.device.load_module_from_source(f"postprocess{module_hash:016x}", builder.build())
        
        self.main_program = self.device.link_program([self.module], [self.module.entry_point("main")])
        self.main_pipeline = self.device.create_compute_pipeline(self.main_program)

        self.backwards_program = self.device.link_program([self.module], [self.module.entry_point("backwards")])
        self.backwards_pipeline = self.device.create_compute_pipeline(self.backwards_program)

    def apply(
        self,
        command_encoder: spy.CommandEncoder,
        input: spy.Texture,
        output: spy.Texture,
    ):
        compute_pass = command_encoder.begin_compute_pass()
        shader_object = compute_pass.bind_pipeline(self.main_pipeline)
        cursor = spy.ShaderCursor(shader_object)

        for i, stage in enumerate(self.stages):
            stage.bind(cursor[f"stage{i}"])

        entry_cursor = cursor.find_entry_point(0)
        entry_cursor.input = input
        entry_cursor.output = output

        compute_pass.dispatch([input.width, input.height, 1])
        compute_pass.end()

    def backwards(
        self,
        command_encoder: spy.CommandEncoder,
        input: spy.Texture,
        weight: spy.Texture,
        output: spy.Texture,
    ):
        compute_pass = command_encoder.begin_compute_pass()
        shader_object = compute_pass.bind_pipeline(self.backwards_pipeline)
        cursor = spy.ShaderCursor(shader_object)

        for i, stage in enumerate(self.stages):
            stage.bind(cursor[f"stage{i}"])

        entry_cursor = cursor.find_entry_point(0)
        entry_cursor.input = input
        entry_cursor.weight = weight
        entry_cursor.gradient = output

        compute_pass.dispatch([input.width, input.height, 1])
        compute_pass.end()

class Exposure(PostProcessStage):
    def __init__(self, stops: float):
        super().__init__("Exposure")

        self.stops = stops

    def bind(self, cursor: spy.ShaderCursor):
        cursor.scale = 2**self.stops

class Tonemapper(PostProcessStage):
    def __init__(self):
        super().__init__("Tonemapper")

    def bind(self, cursor: spy.ShaderCursor): 
        pass


class SrgbEncoder(PostProcessStage):
    def __init__(self, gamma: float):
        super().__init__("SrgbEncoder")
        self.gamma = gamma

    def bind(self, cursor: spy.ShaderCursor): 
        cursor.inv_gamma = 1.0 / self.gamma
