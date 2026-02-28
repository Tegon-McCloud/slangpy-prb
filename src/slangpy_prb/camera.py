import slangpy as spy

from . import Transform

class PerspectiveCamera:
    def __init__(
        self,
        transform: Transform = Transform.identity(),
        vfov: float = 1.0,
        aspect_ratio: float = 1.0,
    ):
        super().__init__()
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

