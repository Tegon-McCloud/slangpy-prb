import pathlib
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import slangpy as spy

from tqdm import tqdm
import imageio.v3 as iio
import matplotlib.pyplot as plt

from . import *

def save_img(img: npt.NDArray, filename: str):
    img = np.clip(img, 0.0, 1.0)
    img = (img * 255).astype(np.uint8)
    
    iio.imwrite(filename, img) 

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

    renderer = PathTracer(device, shader_table_builder)
    backpropagater = ReplayBackpropagater(device, shader_table_builder)
    loss = L2Loss(device, reference)
    optimizer = Adam(
        device,
        scene.variables,
        scene.gradient,
        learning_rate=0.1,
    )
    # optimizer = GradientDescent(
    #     device,
    #     scene.variables,
    #     scene.gradient,
    #     learning_rate=0.5,
    # )

    adjoint = device.create_texture(
        format=spy.Format.rgba32_float,
        width=width,
        height=height,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
        label="adjoint",
    )

    error_target = device.create_texture(
        format=spy.Format.rgba32_float,
        width=width,
        height=height,
        usage=spy.TextureUsage.unordered_access,
        label="error_target",
    )

    n = 100
    losses = np.empty(shape=(n,), dtype=np.float32)
    
    for iteration in tqdm(range(n)):
        primal = renderer.render(
            scene,
            width,
            height,
            sample_count=1 << 10,
        )

        losses[iteration] = loss.loss(primal)

        command_encoder = device.create_command_encoder()
        scene.zero_grad(command_encoder)
        loss.adjoint(command_encoder, primal, adjoint)
        device.submit_command_buffer(command_encoder.finish())

        backpropagater.backpropagate(
            scene,
            adjoint,
            sample_count=1 << 10,
            error_target=error_target,
        )
        save_img(error_target.to_numpy()[:,:,0:3], f"./output/error.png")

        command_encoder = device.create_command_encoder()
        optimizer.step(command_encoder)
        device.submit_command_buffer(command_encoder.finish())
        
        if iteration % 5 == 0:
            values = scene.variables.parameter_buffer.to_numpy().view(np.float32)
            tqdm.write(f"parameters: {values}")
            save_img(tonemap(primal.device, primal).to_numpy(), f"./output/primal_{iteration:02}.png")

    with open("output/loss_over_iteration.npy", "wb") as f:
        np.save(f, losses)
    scene.download()

def loss_over_roughness(
    device: spy.Device,
    width: int,
    height: int,
    stage: Stage,
): 
    
    # material = Metals.gold(roughness=0.4, requires_grad=True)
    # variable_index = 6
    material = Material.microfacet_dielectric_ss(ior=1.5, roughness=(0.4, True))
    variable_index = 0
    stage.replace_material(MaterialId(0), material)

    shader_table_builder = ShaderTableBuilder()
    scene = Scene(device, stage, shader_table_builder)
    path_tracer = PathTracer(device, shader_table_builder)
    backpropagater = ReplayBackpropagater(device, shader_table_builder)

    reference = path_tracer.render(
        scene,
        width,
        height,
        sample_count=1 << 10,
        seed=1234,
    )

    loss = L2Loss(device, reference)

    adjoint = device.create_texture(
        format=spy.Format.rgba32_float,
        width=width,
        height=height,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
        label="adjoint",
    )

    error_target = device.create_texture(
        format=spy.Format.rgba32_float,
        width=width,
        height=height,
        usage=spy.TextureUsage.unordered_access,
        label="error_target",
    )

    n = 100

    values = np.linspace(0.01, 1.0, num=n, endpoint=True, dtype=np.float32)
    losses = np.empty(shape=n, dtype=np.float32)
    gradients = np.empty(shape=n, dtype=np.float32)

    for i in tqdm(range(n)):
        
        variables = scene.variables.parameter_buffer.to_numpy().view(np.float32)
        variables[variable_index] = values[i]
        scene.variables.parameter_buffer.copy_from_numpy(variables)
    
        primal = path_tracer.render(
            scene,
            width=reference.width,
            height=reference.height,
            sample_count=1 << 10,
        )

        losses[i] = loss.loss(primal)

        command_encoder = device.create_command_encoder()
        scene.zero_grad(command_encoder)
        loss.adjoint(command_encoder, primal, adjoint)
        device.submit_command_buffer(command_encoder.finish())
        
        backpropagater.backpropagate(
            scene,
            adjoint,
            sample_count=1 << 10,
            error_target=error_target,
        )
        save_img(error_target.to_numpy()[:,:,0:3], f"./output/error.png")

        scene_gradient = scene.gradient.parameter_buffer.to_numpy().view(np.float32)
        gradients[i] = scene_gradient[variable_index]

    with open("output/loss_over_roughness.npz", "wb") as f:
        np.savez(f, values=values, losses=losses, gradients=gradients)


def bsdf_scatter(device: spy.Device):
    module = spy.Module.load_from_file(device, "shaders/bsdf.slang")

    n = 1 << 9
    theta = 1.6
    phi = -np.pi / 4.0

    wo = spy.float3(np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta))
    wi = spy.NDBuffer(device, shape=(n,), dtype=spy.float3)
    pdf = np.empty(shape=(n,), dtype=np.float32)

    rng = np.random.default_rng(seed=12345)
    rand_state = rng.integers(0xffffffff, endpoint=True, size=(n,), dtype=np.uint32)
    
    bsdf_buffer: spy.NDBuffer = module.sample_microfacet_dielectric_ss(
        wo,
        wi,
        pdf,
        rand_state,
        1.5,
        0.4,
    )

    wi = wi.to_numpy()
    bsdf = bsdf_buffer.to_numpy()

    mask = (wi[:,0] != 0.0) | (wi[:,1] != 0.0) | (wi[:,2] != 0.0)
    wi = wi[mask,:]
    pdf = pdf[mask]
    bsdf = bsdf[mask]

    bsdf_eval_buffer: spy.NDBuffer = module.evaluate_microfacet_dielectric_ss(
        wo,
        wi,
        1.5,
        0.4,
    )

    bsdf_eval = bsdf_eval_buffer.to_numpy()

    error = (bsdf - bsdf_eval) / bsdf

    print(np.median(error[~np.isnan(error)]))

    mean_bsdf = np.mean(bsdf, axis=1)
    weight = mean_bsdf / pdf

    print(wi.shape)
    print(np.max(weight))

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-1.0, 1.0)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$b$")
    ax.set_zlabel("$n$")
    
    ax.plot([0.0, wo.x], [0.0, wo.y], [0.0, wo.z], color='black')
    ax.scatter(wi[:,0], wi[:,1], cast(Any, wi[:,2]), c=weight/np.max(weight), cmap='jet')

    fig.savefig("output/bsdf_samples.pdf")


def main():
    device = spy.create_device(
        include_paths=[pathlib.Path("./shaders")],
        enable_debug_layers=True,
        enable_print=True,
        type=spy.DeviceType.vulkan,
    )
    
    # bsdf_scatter(device)

    width = 960
    height = 540

    stage = Stage()

    stage.load_gltf("./assets/XYZRGBDragon.glb")

    environment_image = iio.imread("assets/kloppenheim_06_puresky_4k.hdr")
    environment_id = stage.add_texture(Texture(environment_image))
    stage.set_environment(environment_id)

    # plot_loss(device, width, height, stage)

    # stage.replace_material(0, Material.lambertian(color=spy.float3(0.8, 0.2, 0.2)))
    stage.replace_material(MaterialId(0), Metals.gold(roughness=0.4))
    # stage.replace_material(MaterialId(0), Material.microfacet_dielectric_ss(ior=1.5, roughness=0.4))

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
    stage.replace_material(MaterialId(0), Metals.copper(roughness=0.8, requires_grad=True))
    # stage.replace_material(0, Material.microfacet_dielectric_ss(ior=1.5, roughness=(0.5, True)))

    optimize(
        device,
        reference,
        stage,
    )

    values = [p.value for p in stage.get_material(MaterialId(0)).parameters]
    print(f"parameters: {values}")


if __name__ == "__main__":
    main()
