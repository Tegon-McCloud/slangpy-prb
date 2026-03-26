import numpy as np
import numpy.typing as npt
import slangpy as spy

class Texture:
    def __init__(
        self,
        image: npt.NDArray,
    ):
        super().__init__()

        if image.shape[2] == 3:
            alpha = np.full(
                shape=(image.shape[0], image.shape[1], 1),
                fill_value=Texture._alpha_value_lookup[image.dtype],
                dtype=image.dtype,
            )
            image = np.concatenate([image, alpha], axis=2)

        self.image = image

    _alpha_value_lookup = {
        np.dtype(np.uint8): 255,
        np.dtype(np.float32): 1.0,
    }

    _format_lookup = {
        (1, np.dtype(np.uint8)): spy.Format.r8_unorm,
        (2, np.dtype(np.uint8)): spy.Format.rg8_unorm,
        (4, np.dtype(np.uint8)): spy.Format.rgba8_unorm,

        (1, np.dtype(np.float32)): spy.Format.r32_float,
        (2, np.dtype(np.float32)): spy.Format.rg32_float,  
        (4, np.dtype(np.float32)): spy.Format.rgba32_float,
    }

    @property
    def format(self) -> spy.Format:
        return Texture._format_lookup[(self.image.shape[2], self.image.dtype)]

    @property
    def width(self) -> int:
        return self.image.shape[1]

    @property
    def height(self) -> int:
        return self.image.shape[0]

# class Texture:
#     def __init__(
#         self,
#         bitmap: spy.Bitmap,
#     ):
#         self.bitmap = bitmap

