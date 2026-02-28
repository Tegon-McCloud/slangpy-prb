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
    def quad() -> 'Mesh':
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
        ], dtype=np.float32)

        indices = np.array([
            [0, 1, 2],
            [3, 2, 1],
        ], dtype=np.uint32)

        return Mesh(positions=vertices, normals=normals, indices=indices)

    def sphere_uv(n: int = 8) -> 'Mesh':
        
        m = 2 * n
        positions = np.empty(shape=(m*(n+1), 3), dtype=np.float32)

        phi = np.linspace(0, 2.0 * np.pi, m, endpoint=False, dtype=np.float32)[np.newaxis,:]
        theta = np.linspace(0, np.pi, n + 1, dtype=np.float32)[:,np.newaxis]

        positions[:,0] = (np.cos(phi) * np.sin(theta)).flatten()
        positions[:,1] = (np.sin(phi) * np.sin(theta)).flatten()
        positions[:,2] = np.broadcast_to(np.cos(theta), shape=np.broadcast_shapes(phi.shape, theta.shape)).flatten()

        normals = np.copy(positions)

        i = np.repeat(np.arange(0, n, dtype=np.uint32), m)
        j = np.tile(np.arange(0, m, dtype=np.uint32), n)

        indices = np.empty(shape=(2 * n * m, 3), dtype=np.uint32)

        linearize = lambda i, j: i * m + j % m

        indices[0::2,0] = linearize(i, j)
        indices[0::2,1] = linearize(i + 1, j)
        indices[0::2,2] = linearize(i, j + 1)

        indices[1::2,0] = linearize(i, j + 1)
        indices[1::2,1] = linearize(i + 1, j)
        indices[1::2,2] = linearize(i + 1, j + 1)

        return Mesh(positions=positions, normals=normals, indices=indices)




