from dataclasses import dataclass

@dataclass(frozen=True)
class VariableId:
    index: int

@dataclass(frozen=True)
class TextureId:
    index: int

@dataclass(frozen=True)
class MeshId:
    index: int

@dataclass(frozen=True)
class MaterialId:
    index: int

@dataclass(frozen=True)
class InstanceId:
    index: int