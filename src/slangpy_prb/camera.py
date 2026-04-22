import slangpy as spy

from . import Transform, CodeBuilder

class Camera:
    def __init__(self, module_name: str):
        super().__init__()
        self.module_name = module_name

    def bind(self, cursor: spy.ShaderCursor): ...

class PerspectiveCamera(Camera):
    def __init__(
        self,
        transform: Transform = Transform.identity(),
        vfov: float = 1.0,
        aspect_ratio: float = 1.0,
    ):
        super().__init__("perspective_camera.slang")
        self.transform = transform
        self.vfov = vfov
        self.aspect_ratio = aspect_ratio

    def bind(self, cursor: spy.ShaderCursor):
        matrix = self.transform.to_matrix()
        y_scale = spy.math.tan(0.5 * self.vfov)
        x_scale = self.aspect_ratio * y_scale

        cursor.horizontal = matrix.get_col(0) * x_scale
        cursor.vertical = matrix.get_col(1) * y_scale
        cursor.forward = -matrix.get_col(2)
        cursor.position = matrix.get_col(3)

class MatrixCamera(Camera):
    def __init__(
        self,
        inv_intrinsic_matrix: spy.float3x3,
        resolution: spy.uint2,
        transform: Transform = Transform.identity(),
    ):
        super().__init__("matrix_camera.slang")
        self.inv_intrinsic_matrix = inv_intrinsic_matrix
        self.resolution = resolution
        self.transform = transform

    def bind(self, cursor: spy.ShaderCursor):
        
        matrix = spy.math.matrix_from_quat(self.transform.rotation)
        matrix = spy.math.mul(spy.math.transpose(matrix), self.inv_intrinsic_matrix)
        matrix.set_col(0, matrix.get_col(0) * self.resolution.x)
        matrix.set_col(1, matrix.get_col(1) * self.resolution.y) 

        cursor.position = self.transform.translation
        cursor.matrix = matrix


