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

class MicrofacetMaterial(Material):
    def __init__(
        self,
        roughness: float,
    ):
        super().__init__(
            evaluate_entry_point="call_evaluate_microfacet",
            sample_entry_point="call_sample_microfacet",
        )

        self.roughness = roughness

    def pack_parameters(self) -> bytes:
        return struct.pack(
            "f",
            self.roughness,
        )