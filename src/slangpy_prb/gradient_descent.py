from typing import Callable

import slangpy as spy
from tqdm import tqdm
import numpy as np
import numpy.typing as npt
from PIL import Image

from . import Scene, PathTracer, ReplayBackpropagater

def save_img(img: npt.NDArray, filename: str):
    img = np.clip(img, 0.0, 1.0)
    img = (img * 255).astype(np.uint8)
    
    Image.fromarray(img).save(filename) 

class GradientDescent:

    def __init__(
        self,
        device: spy.Device,
    ):
        super().__init__()

        self.device = device

    def optimize(
        self,
        scene: Scene,
        adjoint_callback: Callable[[spy.CommandEncoder, spy.Texture, spy.Texture], None],
        width: int,
        height: int,
        renderer: PathTracer,
        backpropagater: ReplayBackpropagater,
        iteration_callback: Callable[[int, Scene, spy.Texture], None] | None,
    ):
        adjoint = self.device.create_texture(
            format=spy.Format.rgba32_float,
            width=width,
            height=height,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
            label="adjoint",
        )

        error_target = self.device.create_texture(
            format=spy.Format.rgba32_float,
            width=width,
            height=height,
            usage=spy.TextureUsage.unordered_access,
            label="error_target",
        )
        
        for iteration in tqdm(range(100)):

            primal = renderer.render(
                scene,
                width,
                height,
                sample_count=1 << 10,
            )

            command_encoder = self.device.create_command_encoder()
            adjoint_callback(command_encoder, primal, adjoint)
            scene.zero_grad(command_encoder)
            self.device.submit_command_buffer(command_encoder.finish())

            backpropagater.backpropagate(
                scene,
                adjoint,
                sample_count=1 << 10,
                error_target=error_target,
            )

            value = scene.materials.variables_buffer.to_numpy().view(np.float32)
            gradient = scene.materials.gradient_buffer.to_numpy().view(np.float32)

            tqdm.write(f"value: {value}")
            tqdm.write(f"gradient: {gradient}")

            value -= 0.5 * gradient

            scene.materials.variables_buffer.copy_from_numpy(value)

            save_img(error_target.to_numpy()[:,:,0:3], f"./output/error.png")

            if iteration_callback != None:
                iteration_callback(iteration, scene, primal)

