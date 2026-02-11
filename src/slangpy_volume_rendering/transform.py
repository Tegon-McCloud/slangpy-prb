import slangpy as spy

class Transform:
    def __init__(self):
        super().__init__()

        self.translation = spy.float3(0)
        self.rotation = spy.quatf.identity()
        self.scale = spy.float3(1)

    @staticmethod
    def identity() -> 'Transform':
        return Transform()

    @staticmethod
    def from_xyz(x: float, y: float, z: float) -> 'Transform':
        transform = Transform()
        transform.translation = spy.float3(x, y, z)
        
        return transform

    def rotate_x(self, angle: float):
        half_angle = 0.5 * angle
        rotation = spy.quatf(spy.math.sin(half_angle), 0.0, 0.0, spy.math.cos(half_angle))
        self.rotation = spy.math.mul(rotation, self.rotation)

    def rotate_y(self, angle: float):
        half_angle = 0.5 * angle
        rotation = spy.quatf(0.0, spy.math.sin(half_angle), 0.0, spy.math.cos(half_angle))
        self.rotation = spy.math.mul(rotation, self.rotation)
    
    def rotate_z(self, angle: float):
        half_angle = 0.5 * angle
        rotation = spy.quatf(0.0, 0.0, spy.math.sin(half_angle), spy.math.cos(half_angle))
        self.rotation = spy.math.mul(rotation, self.rotation)

    def translate(self, vector: spy.float3):
        self.translation = self.translation + vector

    def to_matrix(self) -> spy.float3x4:
        rotation_matrix = spy.math.matrix_from_quat(self.rotation)

        matrix = spy.float3x4()
        matrix.set_col(0, rotation_matrix.get_col(0) * self.scale.x)
        matrix.set_col(1, rotation_matrix.get_col(1) * self.scale.y)
        matrix.set_col(2, rotation_matrix.get_col(2) * self.scale.z)
        matrix.set_col(3, self.translation)

        return matrix

    def transform_vector(self, vector: spy.float3) -> spy.float3:
        vector = self.scale * vector
        vector = self.rotation * vector
        return vector

    def transform_point(self, point: spy.float3) -> spy.float3:
        point = self.scale * point
        point = self.rotation * point
        point += self.translation
        return point