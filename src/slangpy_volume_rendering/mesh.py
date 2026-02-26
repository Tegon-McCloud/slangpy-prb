import numpy as np
import numpy.typing as npt

class Mesh:
    def __init__(
        self,
        positions: npt.NDArray[np.float32],
        normals: npt.NDArray[np.float32],
        indices: npt.NDArray[np.uint32],
    ):
        super().__init__()
        self.positions = positions
        self.normals = normals
        self.indices = indices

    @property
    def vertex_count(self):
        return self.positions.shape[0]

    @property
    def triangle_count(self):
        return self.indices.shape[0]

    @property
    def index_count(self):
        return self.triangle_count * 3

    @staticmethod
    def quad():
        vertices = np.array([
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ], dtype=np.float32)

        normals = np.array([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ])

        indices = np.array([
            [0, 1, 2],
            [3, 2, 1],
        ], dtype=np.uint32)

        return Mesh(positions=vertices, normals=normals, indices=indices)
