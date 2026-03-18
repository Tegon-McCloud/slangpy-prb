import struct
import math
from typing import Any
from dataclasses import dataclass, field

import slangpy as spy

@dataclass(unsafe_hash=True)
class MaterialParameter:
    name: str
    value: float = field(hash=False)
    requires_grad: bool
    range: tuple[float, float] = field(hash=False)

    def pack(self) -> bytes:
        return struct.pack("f", self.value)

@dataclass(unsafe_hash=True)
class Material:
    parameters: tuple[MaterialParameter, ...]
    evaluate_fn_name: str
    sample_fn_name: str
    
    def _parameter_loads(self) -> str:
        constant_counter = 0
        variable_counter = 0

        loads = ""

        for parameter in self.parameters:
            if parameter.requires_grad:
                loads += f"scene.materials.variables[io.variables_start_index + {variable_counter}],\n"
                variable_counter += 1
            else:
                loads += f"scene.materials.constants[io.constants_start_index + {constant_counter}],\n"
                constant_counter += 1

        return loads


    def evaluate_shader(self, entry_point: str) ->  str:
        
        loads = self._parameter_loads()

        return f"""
        [shader("callable")]
        void {entry_point}(inout BsdfEvaluation io) {{
            io.value = {self.evaluate_fn_name}(
                io.wo,
                io.wi,
                {loads}
            );
        }}
        """
    
    def sample_shader(self, entry_point: str) -> str:

        loads = self._parameter_loads()

        return f"""
        [shader("callable")]
        void {entry_point}(inout BsdfSample io) {{
            io.value = {self.sample_fn_name}(
                io.wo,
                io.wi,
                io.pdf,
                io.rand_state,
                {loads}
            );
        }}
        """

    def backpropagate_shader(self, entry_point: str) -> str:
        constant_counter = 0
        variable_counter = 0

        loads = ""
        arguments = ""
        stores = ""

        for parameter in self.parameters:
            if parameter.requires_grad:
                loads += f"DifferentialPair<float> {parameter.name} = diffPair(scene.materials.variables[io.variables_start_index + {variable_counter}]);\n"
                stores += f"""
                if (isfinite({parameter.name}.d)) {{
                    float {parameter.name}_wave = WaveActiveSum({parameter.name}.d);

                    if (WaveIsFirstLane()) {{
                        scene.materials.gradient[io.variables_start_index + {variable_counter}].add({parameter.name}_wave);
                    }}
                }}
                """
                variable_counter += 1
            else:
                loads += f"DifferentialPair<float> {parameter.name} = diffPair(scene.materials.constants[io.constants_start_index + {constant_counter}]);\n"
                constant_counter += 1
            arguments += f"{parameter.name},\n"

        return f"""
        [shader("callable")]
        void {entry_point}(inout BsdfBackpropagation io) {{
        
            {loads}
            bwd_diff({self.evaluate_fn_name})(
                io.wo,
                io.wi,
                {arguments}
                io.weight,
            );

            {stores}
        }}
        """
    
    @staticmethod
    def _standardize_parameter(
        x: Any | tuple[Any, bool],
    ) -> tuple[Any, bool]:
        if isinstance(x, tuple) and len(x) == 2 and isinstance(x[1], bool):
            return x
        return (x, False)

    @staticmethod
    def lambertian(
        color: spy.float3 | tuple[spy.float3, bool],
    ) -> 'Material':
        color, color_requires_grad = Material._standardize_parameter(color)
        
        return Material(
            parameters=(
                MaterialParameter("color_r", color.x, color_requires_grad, (0.0, 1.0)),
                MaterialParameter("color_g", color.y, color_requires_grad, (0.0, 1.0)),
                MaterialParameter("color_b", color.z, color_requires_grad, (0.0, 1.0)),
            ),
            evaluate_fn_name="evaluate_lambertian",
            sample_fn_name="sample_lambertian",
        )

    @staticmethod
    def microfacet_conductor_ss(
        ior: spy.float3,
        extinction: spy.float3,
        roughness: float,
        requires_grad: bool = False,
    ) -> 'Material':
        return Material(
            parameters=(
                MaterialParameter("ior_r", ior.x, requires_grad, (0.0, math.inf)),
                MaterialParameter("ior_g", ior.y, requires_grad, (0.0, math.inf)),
                MaterialParameter("ior_b", ior.z, requires_grad, (0.0, math.inf)),
                MaterialParameter("extinction_r", extinction.x, requires_grad, (0.0, math.inf)),
                MaterialParameter("extinction_g", extinction.y, requires_grad, (0.0, math.inf)),
                MaterialParameter("extinction_b", extinction.z, requires_grad, (0.0, math.inf)),
                MaterialParameter("roughness", roughness, requires_grad, (0.0, 1.0)),
            ),
            evaluate_fn_name="evaluate_microfacet_conductor_ss",
            sample_fn_name="sample_microfacet_conductor_ss",
        )

    @staticmethod
    def microfacet_dielectric_ss(
        ior: float | tuple[float, bool],
        roughness: float | tuple[float, bool],
    ) -> 'Material':
        ior, ior_requires_grad = Material._standardize_parameter(ior)
        roughness, roughness_requires_grad = Material._standardize_parameter(roughness)

        return Material(
            parameters=(
                MaterialParameter("ior", ior, ior_requires_grad, (0.0, math.inf)),
                MaterialParameter("roughness", roughness, roughness_requires_grad, (0.0, 1.0)),
            ),
            evaluate_fn_name="evaluate_microfacet_dielectric_ss",
            sample_fn_name="sample_microfacet_dielectric_ss",
        )

class Metals:
    @staticmethod
    def copper(roughness: float, requires_grad: bool = False) -> Material:
        return Material.microfacet_conductor_ss(
            ior=spy.float3(0.27527, 1.0066, 1.2444),
            extinction=spy.float3(3.3726, 2.5823, 2.4352),
            roughness=roughness,
            requires_grad=requires_grad,
        )

    @staticmethod
    def gold(roughness: float, requires_grad: bool = False) -> Material:
        return Material.microfacet_conductor_ss(
            ior=spy.float3(0.18836, 0.42415, 1.3489),
            extinction=spy.float3(3.4034, 2.4721, 1.8851),
            roughness=roughness,
            requires_grad=requires_grad,
        )

    @staticmethod
    def silver(roughness: float, requires_grad: bool = False) -> Material:
        return Material.microfacet_conductor_ss(
            ior=spy.float3(0.056909, 0.0595825, 0.044439),
            extinction=spy.float3(4.2543, 3.5974, 2.7511),
            roughness=roughness,
            requires_grad=requires_grad,
        )

    @staticmethod 
    def aluminium(roughness: float, requires_grad: bool = False) -> Material:
        return Material.microfacet_conductor_ss(
            ior=spy.float3(1.4303, 1.0152, 0.66843),
            extinction=spy.float3(7.5081, 6.6273, 5.5748),
            roughness=roughness,
            requires_grad=requires_grad,
        )

    @staticmethod 
    def cobalt(roughness: float, requires_grad: bool = False) -> Material:
        return Material.microfacet_conductor_ss(
            ior=spy.float3(1.7715, 2.0524, 1.7715),
            extinction=spy.float3(3.3385, 3.8242, 3.3385),
            roughness=roughness,
            requires_grad=requires_grad,
        )
