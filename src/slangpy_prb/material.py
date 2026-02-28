import struct

import slangpy as spy

class Material:
    def __init__(
        self,
        evaluate_entry_point: str,
        sample_entry_point: str,
    ):
        super().__init__()

        self.evaluate_entry_point = evaluate_entry_point
        self.sample_entry_point = sample_entry_point

    def pack_parameters(self) -> bytes: ...

class LambertianMaterial(Material):
    def __init__(
        self,
        color: spy.float3,
    ):
        super().__init__(
            evaluate_entry_point="call_evaluate_lambertian",
            sample_entry_point="call_sample_lambertian",
        )

        self.color = color

    def pack_parameters(self) -> bytes:
        return struct.pack(
            "fff",
            self.color.x,
            self.color.y,
            self.color.z,
        )

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

class MicrofacetMaterial(Material):
    def __init__(
        self,
        roughness: float,
        ior: float,
    ):
        super().__init__(
            evaluate_entry_point="call_evaluate_microfacet",
            sample_entry_point="call_sample_microfacet",
        )

        self.roughness = roughness
        self.ior = ior

    def pack_parameters(self) -> bytes:
        return struct.pack(
            "ff",
            self.roughness,
            self.ior,
        )