from typing import TypeAlias
from dataclasses import dataclass

from . import CodeBuilder, VariableId, TextureId

@dataclass(frozen=True)
class TextureChannel:
    id: TextureId
    channel: int

Argument: TypeAlias = float | VariableId | TextureChannel
NonTextureArgument: TypeAlias = float | VariableId

@dataclass(unsafe_hash=True)
class MaterialParameter:
    name: str
    argument: Argument

@dataclass(unsafe_hash=True)
class Material:
    parameters: tuple[MaterialParameter, ...]
    evaluate_fn_name: str
    sample_fn_name: str
    
    def referenced_textures(self) -> list[TextureId]:
        referenced_textures: set[TextureId] = set()

        for parameter in self.parameters:
            match parameter.argument:
                case TextureChannel(id, channel):
                    referenced_textures.add(id)

        return list(referenced_textures)

    def shader(self, entry_point: str, builder: CodeBuilder):
        builder.append_line(f"[shader(\"callable\")]")
        builder.append_line(f"void {entry_point}(inout MaterialCallData data)")
        builder.begin_block()

        for id in self.referenced_textures():

            builder.declare("float4", f"texture{id.index}", f"load_from_uv(scene.textures[{id.index}], data.uv)")

        constant_counter = 0

        for parameter in self.parameters:
            match parameter.argument:
                case float(value):
                    expr = f"scene.materials.constants[data.constants_start_index + {constant_counter}]"
                    constant_counter += 1
                case VariableId(index):
                    expr = f"scene.variables[{index}]"
                case TextureChannel(id, channel):
                    expr = f"texture{id.index}[{channel}]"
                case _:
                    raise RuntimeError("Unknown argument type")

            builder.declare("var", f"{parameter.name}", expr)

        builder.newline()

        builder.append_line(f"if ((data.flags & MaterialCallData.EVALUATE_FLAG) != 0) ")
        builder.begin_block()
        builder.append_line(f"data.evaluation.value = {self.evaluate_fn_name}(")
        builder.inc_indent()
        builder.append_line(f"data.wo,")
        builder.append_line(f"data.evaluation.wi,")

        for parameter in self.parameters:
            builder.append_line(f"{parameter.name},")

        builder.dec_indent()
        builder.append_line(");")
        builder.end_block()
        builder.newline()

        builder.append_line(f"if ((data.flags & MaterialCallData.SAMPLE_FLAG) != 0) ")
        builder.begin_block()
        builder.append_line(f"data.sample = {self.sample_fn_name}(")
        builder.inc_indent()
        builder.append_line(f"data.wo,")
        builder.append_line(f"data.rand_state,")

        for parameter in self.parameters:
            builder.append_line(f"{parameter.name},")

        builder.dec_indent()
        builder.append_line(");")
        builder.end_block()
        builder.end_block()
    
    def backpropagate_shader(self, entry_point: str, builder: CodeBuilder):
        builder.append_line(f"[shader(\"callable\")]")
        builder.append_line(f"void {entry_point}(inout MaterialBackpropagation data)")
        builder.begin_block()

        for id in self.referenced_textures():
            builder.declare("float4", f"texture{id.index}", f"scene.textures[{id.index}].Sample(scene.default_sampler, data.uv)")

        constant_counter = 0

        for parameter in self.parameters:
            match parameter.argument:
                case float(value):
                    expr = f"scene.materials.constants[data.constants_start_index + {constant_counter}]"
                    constant_counter += 1
                case VariableId(index):
                    expr = f"scene.variables[{index}]"
                case TextureChannel(id, channel):
                    expr = f"texture{id.index}[{channel}]"
                case _:
                    raise RuntimeError("Unknown argument type")

            builder.declare("var", f"{parameter.name}", f"diffPair({expr})")
        builder.newline()

        builder.append_line(f"bwd_diff({self.evaluate_fn_name})(")
        builder.inc_indent()
        builder.append_line(f"data.wo,")
        builder.append_line(f"data.wi,")
        
        for parameter in self.parameters:
            builder.append_line(f"{parameter.name},")

        builder.append_line("data.weight,")
        builder.dec_indent()
        builder.append_line(");")

        for parameter in self.parameters:
            match parameter.argument:
                case VariableId(index):
                    builder.append_line(f"if (isfinite({parameter.name}.d))")
                    builder.begin_block()
                    builder.declare("float", f"{parameter.name}_wave", f"WaveActiveSum({parameter.name}.d)")
                    builder.append_line("if (WaveIsFirstLane())")
                    builder.begin_block()
                    builder.append_line(f"scene.gradient[{index}] += {parameter.name}_wave;")
                    builder.end_block()
                    builder.end_block()

        builder.end_block()

    @staticmethod
    def _standardize_float3_argument(argument: tuple[Argument, Argument, Argument] | TextureId) -> tuple[Argument, Argument, Argument]:

        match argument:
            case TextureId():
                return (
                    TextureChannel(argument, 0),
                    TextureChannel(argument, 1),
                    TextureChannel(argument, 2),    
                )
        
        return argument

    @staticmethod
    def lambertian(
        reflectance: tuple[Argument, Argument, Argument] | TextureId,
    ) -> 'Material':
        
        reflectance = Material._standardize_float3_argument(reflectance)

        return Material(
            parameters=(
                MaterialParameter("reflectance_r", reflectance[0]),
                MaterialParameter("reflectance_g", reflectance[1]),
                MaterialParameter("reflectance_b", reflectance[2]),
            ),
            evaluate_fn_name="evaluate_lambertian",
            sample_fn_name="sample_lambertian",
        )

    @staticmethod
    def microfacet_conductor_ss(
        ior: tuple[Argument, Argument, Argument] | TextureId,
        extinction: tuple[Argument, Argument, Argument] | TextureId,
        roughness: Argument,
    ) -> 'Material':
        
        ior = Material._standardize_float3_argument(ior)
        extinction = Material._standardize_float3_argument(extinction)

        return Material(
            parameters=(
                MaterialParameter("ior_r", ior[0]),
                MaterialParameter("ior_g", ior[1]),
                MaterialParameter("ior_b", ior[2]),
                MaterialParameter("extinction_r", extinction[0]),
                MaterialParameter("extinction_g", extinction[1]),
                MaterialParameter("extinction_b", extinction[2]),
                MaterialParameter("roughness", roughness),
            ),
            evaluate_fn_name="evaluate_microfacet_conductor_ss",
            sample_fn_name="sample_microfacet_conductor_ss",
        )

    @staticmethod
    def microfacet_dielectric_ss(
        ior: NonTextureArgument,
        roughness: Argument,
    ) -> 'Material':
        return Material(
            parameters=(
                MaterialParameter("ior", ior),
                MaterialParameter("roughness", roughness),
            ),
            evaluate_fn_name="evaluate_microfacet_dielectric_ss",
            sample_fn_name="sample_microfacet_dielectric_ss",
        )

class Metals:
    @staticmethod
    def copper(roughness: Argument) -> Material:
        return Material.microfacet_conductor_ss(
            ior=(0.27527, 1.0066, 1.2444),
            extinction=(3.3726, 2.5823, 2.4352),
            roughness=roughness,
        )

    @staticmethod
    def gold(roughness: Argument) -> Material:
        return Material.microfacet_conductor_ss(
            ior=(0.18836, 0.42415, 1.3489),
            extinction=(3.4034, 2.4721, 1.8851),
            roughness=roughness,
        )

    @staticmethod
    def silver(roughness: Argument) -> Material:
        return Material.microfacet_conductor_ss(
            ior=(0.056909, 0.0595825, 0.044439),
            extinction=(4.2543, 3.5974, 2.7511),
            roughness=roughness,
        )

    @staticmethod 
    def aluminium(roughness: Argument) -> Material:
        return Material.microfacet_conductor_ss(
            ior=(1.4303, 1.0152, 0.66843),
            extinction=(7.5081, 6.6273, 5.5748),
            roughness=roughness,
        )

    @staticmethod 
    def cobalt(roughness: Argument) -> Material:
        return Material.microfacet_conductor_ss(
            ior=(1.7715, 2.0524, 1.7715),
            extinction=(3.3385, 3.8242, 3.3385),
            roughness=roughness,
        )
