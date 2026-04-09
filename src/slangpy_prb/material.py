from typing import TypeAlias
from dataclasses import dataclass

from . import CodeBuilder, VariableId, TextureId

Argument: TypeAlias = float | VariableId | TextureId
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
    
    def shader(self, entry_point: str, builder: CodeBuilder):
        builder.append_line(f"[shader(\"callable\")]")
        builder.append_line(f"void {entry_point}(inout MaterialCallData data)")
        builder.begin_block()

        constant_counter = 0
        variable_counter = 0

        for parameter in self.parameters:
            match parameter.argument:
                case float(value):
                    expr = f"scene.materials.constants[data.constants_start_index + {constant_counter}]"
                    constant_counter += 1
                case VariableId(index):
                    expr = f"scene.variables[{index}]"
                    variable_counter += 1
                case TextureId(index):
                    expr = f"scene.textures[{index}].Sample()"
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

        constant_counter = 0

        for parameter in self.parameters:
            match parameter.argument:
                case float(value):
                    expr = f"scene.materials.constants[data.constants_start_index + {constant_counter}]"
                    constant_counter += 1
                case VariableId(index):
                    expr = f"scene.variables[{index}]"
                case TextureId(index):
                    expr = f"scene.textures[{index}].Sample"
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
    def lambertian(
        reflectance_r: Argument,
        reflectance_g: Argument,
        reflectance_b: Argument,
    ) -> 'Material':
        
        return Material(
            parameters=(
                MaterialParameter("reflectance_r", reflectance_r),
                MaterialParameter("reflectance_g", reflectance_g),
                MaterialParameter("reflectance_b", reflectance_b),
            ),
            evaluate_fn_name="evaluate_lambertian",
            sample_fn_name="sample_lambertian",
        )

    # @staticmethod
    # def microfacet_conductor_ss(
    #     ior: spy.float3,
    #     extinction: spy.float3,
    #     roughness: float,
    #     requires_grad: bool = False,
    # ) -> 'Material':
    #     return Material(
    #         parameters=(
    #             MaterialParameter("ior_r", ior.x, requires_grad, (0.0, math.inf)),
    #             MaterialParameter("ior_g", ior.y, requires_grad, (0.0, math.inf)),
    #             MaterialParameter("ior_b", ior.z, requires_grad, (0.0, math.inf)),
    #             MaterialParameter("extinction_r", extinction.x, requires_grad, (0.0, math.inf)),
    #             MaterialParameter("extinction_g", extinction.y, requires_grad, (0.0, math.inf)),
    #             MaterialParameter("extinction_b", extinction.z, requires_grad, (0.0, math.inf)),
    #             MaterialParameter("roughness", roughness, requires_grad, (0.0, 1.0)),
    #         ),
    #         evaluate_fn_name="evaluate_microfacet_conductor_ss",
    #         sample_fn_name="sample_microfacet_conductor_ss",
    #     )

    # @staticmethod
    # def microfacet_dielectric_ss(
    #     ior: float | tuple[float, bool],
    #     roughness: float | tuple[float, bool],
    # ) -> 'Material':
    #     ior, ior_requires_grad = Material._standardize_parameter(ior)
    #     roughness, roughness_requires_grad = Material._standardize_parameter(roughness)

    #     return Material(
    #         parameters=(
    #             MaterialParameter("ior", ior, ior_requires_grad, (0.0, math.inf)),
    #             MaterialParameter("roughness", roughness, roughness_requires_grad, (0.0, 1.0)),
    #         ),
    #         evaluate_fn_name="evaluate_microfacet_dielectric_ss",
    #         sample_fn_name="sample_microfacet_dielectric_ss",
    #     )

# class Metals:
#     @staticmethod
#     def copper(roughness: float, requires_grad: bool = False) -> Material:
#         return Material.microfacet_conductor_ss(
#             ior=spy.float3(0.27527, 1.0066, 1.2444),
#             extinction=spy.float3(3.3726, 2.5823, 2.4352),
#             roughness=roughness,
#             requires_grad=requires_grad,
#         )

#     @staticmethod
#     def gold(roughness: float, requires_grad: bool = False) -> Material:
#         return Material.microfacet_conductor_ss(
#             ior=spy.float3(0.18836, 0.42415, 1.3489),
#             extinction=spy.float3(3.4034, 2.4721, 1.8851),
#             roughness=roughness,
#             requires_grad=requires_grad,
#         )

#     @staticmethod
#     def silver(roughness: float, requires_grad: bool = False) -> Material:
#         return Material.microfacet_conductor_ss(
#             ior=spy.float3(0.056909, 0.0595825, 0.044439),
#             extinction=spy.float3(4.2543, 3.5974, 2.7511),
#             roughness=roughness,
#             requires_grad=requires_grad,
#         )

#     @staticmethod 
#     def aluminium(roughness: float, requires_grad: bool = False) -> Material:
#         return Material.microfacet_conductor_ss(
#             ior=spy.float3(1.4303, 1.0152, 0.66843),
#             extinction=spy.float3(7.5081, 6.6273, 5.5748),
#             roughness=roughness,
#             requires_grad=requires_grad,
#         )

#     @staticmethod 
#     def cobalt(roughness: float, requires_grad: bool = False) -> Material:
#         return Material.microfacet_conductor_ss(
#             ior=spy.float3(1.7715, 2.0524, 1.7715),
#             extinction=spy.float3(3.3385, 3.8242, 3.3385),
#             roughness=roughness,
#             requires_grad=requires_grad,
#         )
