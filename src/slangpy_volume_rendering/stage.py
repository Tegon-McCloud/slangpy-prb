import slangpy as spy

from . import Transform, PerspectiveCamera, Mesh, Material

class Instance:
    def __init__(
        self,
        mesh_id: int,
        material_id: int,
        transform: Transform,
    ):
        super().__init__()
        self.mesh_id = mesh_id
        self.material_id = material_id
        self.transform = transform

class Stage:
    def __init__(
        self,
        camera: PerspectiveCamera,
        environment: spy.Bitmap | None,
    ):
        super().__init__()
        self.meshes: list[Mesh] = []
        self.materials: list[Material] = []
        self.instances: list[Instance] = []
        self.camera = camera
        self.environment = environment

    def add_mesh(self, mesh: Mesh) -> int:
        mesh_id = len(self.meshes)
        self.meshes.append(mesh)
        return mesh_id

    def add_material(self, material: Material) -> int:
        material_id = len(self.materials)
        self.materials.append(material)
        return material_id

    def add_instance(self, instance: Instance) -> int:
        instance_id = len(self.instances)
        self.instances.append(instance)
        return instance_id
