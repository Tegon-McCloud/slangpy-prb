import random
import pathlib

import numpy as np
import numpy.typing as npt
import slangpy as spy
from PIL import Image
from tqdm import tqdm
import json

from . import *

class Tonemapper:

    def __init__(
        self,
        device: spy.Device,
    ):
        self.device = device
        self.program = self.device.load_program("tonemap.slang", ["main"])

        self.kernel = self.device.create_compute_kernel(self.program)

    def tonemap(
        self,
        command_encoder: spy.CommandEncoder,
        input: spy.Texture,
        output: spy.Texture,    
    ):
        self.kernel.dispatch(
            thread_count=[output.width, output.height, 1],
            vars={
                "tonemapper": {
                    "input": input,
                    "output": output,
                },
            },
            command_encoder=command_encoder,
        )



def save_img(img: npt.NDArray, filename: str):
    img = np.clip(img, 0.0, 1.0)
    img = (img * 255).astype(np.uint8)
    
    Image.fromarray(img).save(filename) 

def tonemap(device: spy.Device, texture: spy.Texture) -> spy.Texture:
    tonemapper = Tonemapper(device)
    output = device.create_texture(
        format=spy.Format.rgba32_float,
        width=texture.width,
        height=texture.height,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
        label="output",
    )

    command_encoder = device.create_command_encoder()
    tonemapper.tonemap(command_encoder, texture, output)
    device.submit_command_buffer(command_encoder.finish())

    return output

class L2Loss:

    def __init__(
        self,
        device: spy.Device,
        reference: spy.Texture,
    ):
        super().__init__()

        self.device = device
        self.reference = reference

        self.adjoint_program = self.device.load_program("shaders/l2_adjoint.slang", ["main"])
        self.adjoint_kernel = self.device.create_compute_kernel(self.adjoint_program)

        self.scale = 2.0 / (self.reference.width * self.reference.height)

    def adjoint(
        self,
        command_encoder: spy.CommandEncoder,
        primal: spy.Texture,
        out: spy.Texture,
    ):
        self.adjoint_kernel.dispatch(
            thread_count=[self.reference.width, self.reference.height, 1],
            vars={
                "scale": self.scale,
                "primal": primal,
                "reference": self.reference,
                "adjoint": out,
            },
            command_encoder=command_encoder,
        )

def save_callback(iteration: int, scene: Scene, primal: spy.Texture):
    if iteration % 10 == 0: 
        save_img(tonemap(primal.device, primal).to_numpy(), f"./output/primal_{iteration:02}.png")

def optimize(
    device: spy.Device,
    reference: spy.Texture,
    stage: Stage,
):
    
    shader_table_builder = ShaderTableBuilder()
    scene = Scene(device, stage, shader_table_builder)

    path_tracer = PathTracer(device, shader_table_builder)
    backpropagater = ReplayBackpropagater(device, shader_table_builder)
    optimizer = GradientDescent(device)
    loss = L2Loss(device, reference)

    optimizer.optimize(
        scene,
        lambda command_encoder, primal, out: loss.adjoint(command_encoder, primal, out),
        reference.width,
        reference.height,
        path_tracer,
        backpropagater,
        save_callback,
    )

    scene.download()

def plot_loss(
    device: spy.Device,
    reference: spy.Texture,
    stage: Stage,
): 
    width = reference.width
    height = reference.height

    error_target = device.create_texture(
        format=spy.Format.rgba32_float,
        width=width,
        height=height,
        usage=spy.TextureUsage.unordered_access,
        label="error_target",
    )

    reference_arr = reference.to_numpy()
    n = 2

    roughness = np.linspace(0.2, 0.6, num=n, endpoint=True, dtype=np.float32)
    loss = np.empty(shape=n, dtype=np.float32)
    gradient = np.empty(shape=n, dtype=np.float32)

    for i in tqdm(range(n)):
        # stage.replace_material(0, MicrofacetDielectricMaterial(ior=1.5, roughness=roughness[i], requires_grad=True))

        shader_table_builder = ShaderTableBuilder()
        scene = Scene(device, stage, shader_table_builder)
        path_tracer = PathTracer(device, shader_table_builder)
        backpropagater = ReplayBackpropagater(device, shader_table_builder)
    
        primal = path_tracer.render(
            scene,
            width=reference.width,
            height=reference.height,
            sample_count=1 << 10,
        )

        primal_arr = primal.to_numpy()
        loss[i] = np.mean((primal_arr - reference_arr)**2)

        adjoint_arr = 2 * (primal.to_numpy() - reference.to_numpy()) / (width * height)
        adjoint = device.create_texture(
            data=adjoint_arr,
            format=spy.Format.rgba32_float,
            width=width,
            height=height,
            usage=spy.TextureUsage.shader_resource,
            label="adjoint",
        )

        command_encoder = device.create_command_encoder()
        scene.zero_grad(command_encoder)
        device.submit_command_buffer(command_encoder.finish())
        
        backpropagater.backpropagate(
            device,
            scene,
            adjoint,
            backpropagater,
            sample_count=1 << 10,
            error_target=error_target,
        )

        scene_gradient = scene.materials.gradient_buffer.to_numpy().view(np.float32)

        gradient[i] = scene_gradient[1]

    json.dump(
        {
            "roughness": roughness.tolist(),
            "loss": loss.tolist(),
            "gradient": gradient.tolist(),
        },
        open("output/loss_over_roughness.json", 'w'),
    )

def main():
    device = spy.create_device(
        include_paths=[pathlib.Path("./shaders")],
        enable_debug_layers=True,
        enable_print=True,
        type=spy.DeviceType.vulkan,
    )

    width = 960
    height = 540

    stage = Stage(
        environment=spy.Bitmap.load_from_file("./assets/kloppenheim_06_puresky_4k.hdr"),
    )

    stage.load_gltf("./assets/XYZRGBDragon.glb")

    # stage.replace_material(0, Material.lambertian(color=spy.float3(0.8, 0.2, 0.2)))
    stage.replace_material(0, Metals.gold(roughness=0.4))
    # stage.replace_material(0, Material.microfacet_dielectric_ss(ior=1.5, roughness=0.4))

    shader_table_builder = ShaderTableBuilder()
    scene = Scene(device, stage, shader_table_builder)
    path_tracer = PathTracer(device, shader_table_builder)

    reference = path_tracer.render(
        scene,
        width,
        height,
        sample_count=1 << 10,
        seed=1234,
    )

    save_img(tonemap(device, reference).to_numpy(), "./output/reference.png")

    # stage.replace_material(0, Material.lambertian(color=spy.float3(0.5, 0.5, 0.5), color_requires_grad=True))
    # stage.replace_material(0, Material.microfacet_dielectric_ss(ior=1.5, roughness=(0.5, True)))
    stage.replace_material(0, Metals.copper(roughness=0.8, requires_grad=True))



    optimize(
        device,
        reference,
        stage,
    )

    values = [p.value for p in stage.get_material(0).parameters]
    print(f"parameters: {values}")

if __name__ == "__main__":
    main()
