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


def render(
    device: spy.Device,
    scene: Scene,
    path_tracer: PathTracer,
    width: int,
    height: int,
    sample_count: int,
    seed: int | None = None,
) -> spy.Texture:
    
    if seed != None:
        random.seed(seed)

    render_target = device.create_texture(
        format=spy.Format.rgba32_float,
        width=width,
        height=height,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
        label="render_target",
    )

    for sample_index in tqdm(range(sample_count)):
        command_encoder = device.create_command_encoder()

        path_tracer.sample(
            command_encoder,
            scene,
            render_target,
            sample_index,
        )
        
        submit_id = device.submit_command_buffer(command_encoder.finish())
        device.wait_for_submit(submit_id)

    return render_target

def backpropagate(
    device: spy.Device,
    scene: Scene,
    adjoint: spy.Texture,
    backpropagater: ReplayBackpropagater,
    sample_count: int,
    error_target: spy.Texture,
    seed: int | None = None,
):
    if seed != None:
        random.seed(seed)


    for _ in tqdm(range(sample_count)):
        command_encoder = device.create_command_encoder()

        backpropagater.sample(
            command_encoder,
            scene,
            adjoint,
            sample_count,
            error_target,
        )
        
        submit_id = device.submit_command_buffer(command_encoder.finish())
        device.wait_for_submit(submit_id)

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

def optimize(
    device: spy.Device,
    reference: spy.Texture,
    stage: Stage,
):
    width = reference.width
    height = reference.height
    
    shader_table_builder = ShaderTableBuilder()
    scene = Scene(device, stage, shader_table_builder)

    path_tracer = PathTracer(device, shader_table_builder)
    backpropagater = ReplayBackpropagater(device, shader_table_builder)
    
    error_target = device.create_texture(
        format=spy.Format.rgba32_float,
        width=width,
        height=height,
        usage=spy.TextureUsage.unordered_access,
        label="error_target",
    )

    for iteration in tqdm(range(100)):

        primal = render(
            device,
            scene,
            path_tracer,
            width,
            height,
            sample_count=1 << 10,
        )

        if iteration % 5 == 0:
            save_img(tonemap(device, primal).to_numpy(), f"./output/primal_{iteration:02}.png")
        
        reference_arr = reference.to_numpy()
        primal_arr = primal.to_numpy()

        loss = np.mean((primal_arr - reference_arr)**2)
        tqdm.write(f"loss: {loss}")

        adjoint_arr = 2 * (primal.to_numpy() - reference.to_numpy()) / (width * height)
        adjoint = device.create_texture(
            data=adjoint_arr,
            format=spy.Format.rgba32_float,
            width=width,
            height=height,
            usage=spy.TextureUsage.shader_resource,
            label="adjoint",
        )
        if iteration % 5 == 0:
           save_img(width * height * adjoint_arr[:,:,0:3], f"./output/adjoint_{iteration:02}.png")

        command_encoder = device.create_command_encoder()
        scene.zero_grad(command_encoder)
        device.submit_command_buffer(command_encoder.finish())

        backpropagate(
            device,
            scene,
            adjoint,
            backpropagater,
            sample_count=1 << 10,
            error_target=error_target,
        )

        value = scene.materials.variables_buffer.to_numpy().view(np.float32)
        gradient = scene.materials.gradient_buffer.to_numpy().view(np.float32)

        tqdm.write(f"value: {value}")
        tqdm.write(f"gradient: {gradient}")

        value[0:gradient.shape[0]] -= 0.5 * gradient

        scene.materials.variables_buffer.copy_from_numpy(value)

        save_img(error_target.to_numpy()[:,:,0:3], f"./output/error.png")

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
    
        primal = render(
            device,
            scene,
            path_tracer,
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
        
        backpropagate(
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

    reference = render(
        device,
        scene,
        path_tracer,
        width,
        height,
        sample_count=1 << 10,
        seed=1234,
    )

    save_img(tonemap(device, reference).to_numpy(), "./output/reference.png")

    # plot_loss(device, reference, stage)

    # stage.replace_material(0, Material.lambertian(color=spy.float3(0.5, 0.5, 0.5), color_requires_grad=True))
    # stage.replace_material(0, Material.microfacet_dielectric_ss(ior=1.5, roughness=(0.5, True)))
    stage.replace_material(0, Metals.copper(roughness=0.8, requires_grad=True))

    optimize(
        device,
        reference,
        stage,
    )


if __name__ == "__main__":
    main()
