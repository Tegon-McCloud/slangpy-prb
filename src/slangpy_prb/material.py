import struct

import slangpy as spy

class Material:
    def __init__(
        self,
        parameter_struct: struct.Struct,
        evaluate_entry_point: str,
        sample_entry_point: str,
        backpropagate_entry_point: str = "",
        requires_grad: bool = False,
    ):
        super().__init__()

        self.parameter_struct = parameter_struct
        self.evaluate_entry_point = evaluate_entry_point
        self.sample_entry_point = sample_entry_point
        self.backpropagate_entry_point = backpropagate_entry_point
        self.requires_grad = requires_grad

    def pack_parameters(self) -> bytes: ...
    def unpack_parameters(self, parameter_bytes: bytes): ...

class LambertianMaterial(Material):
    def __init__(
        self,
        color: spy.float3,
        requires_grad: bool = False,
    ):
        super().__init__(
            parameter_struct=struct.Struct("fff"),
            evaluate_entry_point="call_evaluate_lambertian",
            sample_entry_point="call_sample_lambertian",
            backpropagate_entry_point="call_backpropagate_lambertian",
            requires_grad=requires_grad,
        )

        self.color = color

    def pack_parameters(self) -> bytes:
        return self.parameter_struct.pack(
            self.color.x,
            self.color.y,
            self.color.z,
        )
    
    def unpack_parameters(self, parameters: bytes):
        self.color.x, self.color.y, self.color.z = self.parameter_struct.unpack(parameters)


class SpecularConductorMaterial(Material):
    def __init__(
        self,
        ior: spy.float3,
        extinction: spy.float3,    
    ):
        super().__init__(
            evaluate_entry_point="call_evaluate_specular_conductor",
            sample_entry_point="call_sample_specular_conductor",
        )

        self.ior = ior
        self.extinction = extinction

    @staticmethod
    def copper() -> 'SpecularConductorMaterial':
        return SpecularConductorMaterial(
            ior=spy.float3(0.27527, 1.0066, 1.2444),
            extinction=spy.float3(3.3726, 2.5823, 2.4352),
        )

    @staticmethod
    def gold() -> 'SpecularConductorMaterial':
        return SpecularConductorMaterial(
            ior=spy.float3(0.18836, 0.42415, 1.3489),
            extinction=spy.float3(3.4034, 2.4721, 1.8851),
        )

    @staticmethod
    def silver() -> 'SpecularConductorMaterial':
        return SpecularConductorMaterial(
            ior=spy.float3(0.056909, 0.0595825, 0.044439),
            extinction=spy.float3(4.2543, 3.5974, 2.7511),
        )

    @staticmethod 
    def aluminium() -> 'SpecularConductorMaterial':
        return SpecularConductorMaterial(
            ior=spy.float3(1.4303, 1.0152, 0.66843),
            extinction=spy.float3(7.5081, 6.6273, 5.5748),
        )

    @staticmethod 
    def cobalt() -> 'SpecularConductorMaterial':
        return SpecularConductorMaterial(
            ior=spy.float3(1.7715, 2.0524, 1.7715),
            extinction=spy.float3(3.3385, 3.8242, 3.3385),
        )

    def normal_reflectance(self) -> spy.float3:
        
        reflectance = spy.float3()

        for i in range(3):
            eta = complex(self.ior[i], self.extinction[i])
            reflectance[i] = abs((eta - 1.0) / (eta + 1.0))**2

        return reflectance

    def pack_parameters(self) -> bytes:
        return struct.pack(
            "ffffff",
            self.ior.x,
            self.ior.y,
            self.ior.z,
            self.extinction.x,
            self.extinction.y,
            self.extinction.z,
        )

class SpecularDielectricMaterial(Material):
    def __init__(
        self,
        ior: float,
    ):
        super().__init__(
            evaluate_entry_point="call_evaluate_specular_dielectric",
            sample_entry_point="call_sample_specular_dielectric",
        )
        self.ior = ior

    def pack_parameters(self) -> bytes:
        return struct.pack(
            "f",
            self.ior,
        )
    
class MicrofacetConductorMaterial(Material):
    def __init__(
        self,
        ior: spy.float3,
        extinction: spy.float3,
        roughness: float,
    ):
        super().__init__(
            evaluate_entry_point="call_evaluate_microfacet_conductor_ss",
            sample_entry_point="call_sample_microfacet_conductor_ss",
        )

        self.ior = ior
        self.extinction = extinction
        self.roughness = roughness

    @staticmethod
    def copper(roughness: float) -> 'MicrofacetConductorMaterial':
        return MicrofacetConductorMaterial(
            ior=spy.float3(0.27527, 1.0066, 1.2444),
            extinction=spy.float3(3.3726, 2.5823, 2.4352),
            roughness=roughness,
        )

    @staticmethod
    def gold(roughness: float) -> 'MicrofacetConductorMaterial':
        return MicrofacetConductorMaterial(
            ior=spy.float3(0.18836, 0.42415, 1.3489),
            extinction=spy.float3(3.4034, 2.4721, 1.8851),
            roughness=roughness,
        )

    @staticmethod
    def silver(roughness: float) -> 'MicrofacetConductorMaterial':
        return MicrofacetConductorMaterial(
            ior=spy.float3(0.056909, 0.0595825, 0.044439),
            extinction=spy.float3(4.2543, 3.5974, 2.7511),
            roughness=roughness,
        )

    @staticmethod 
    def aluminium(roughness: float) -> 'MicrofacetConductorMaterial':
        return MicrofacetConductorMaterial(
            ior=spy.float3(1.4303, 1.0152, 0.66843),
            extinction=spy.float3(7.5081, 6.6273, 5.5748),
            roughness=roughness,
        )

    @staticmethod 
    def cobalt(roughness: float) -> 'MicrofacetConductorMaterial':
        return MicrofacetConductorMaterial(
            ior=spy.float3(1.7715, 2.0524, 1.7715),
            extinction=spy.float3(3.3385, 3.8242, 3.3385),
            roughness=roughness,
        )

    def pack_parameters(self) -> bytes:
        return struct.pack(
            "fffffff",
            self.ior.x,
            self.ior.y,
            self.ior.z,
            self.extinction.x,
            self.extinction.y,
            self.extinction.z,
            self.roughness,
        )


class MicrofacetDielectricMaterial(Material):
    def __init__(
        self,
        ior: float,
        roughness: float,
        requires_grad: bool = False,
    ):
        super().__init__(
            parameter_struct=struct.Struct("ff"),
            evaluate_entry_point="call_evaluate_microfacet_dielectric_ss",
            sample_entry_point="call_sample_microfacet_dielectric_ss",
            backpropagate_entry_point="call_backpropagate_microfacet_dielectric_ss",
            requires_grad=requires_grad,
        )

        self.ior = ior
        self.roughness = roughness

    def pack_parameters(self) -> bytes:
        return self.parameter_struct.pack(
            self.ior,
            self.roughness,
        )
    
    def unpack_parameters(self, parameter_bytes: bytes):
        self.ior, self.roughness = self.parameter_struct.unpack(parameter_bytes)
