import pathlib
import random
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import slangpy as spy

from tqdm import tqdm
import imageio.v3 as iio
import matplotlib.pyplot as plt

from . import *

def save_img(img: npt.NDArray, filename: pathlib.Path, srgb_encode: bool = False):
    img = np.clip(img, 0.0, 1.0)

    if srgb_encode:
        img = np.where(
        img <= 0.0031308,
        12.92 * img,
        1.055 * np.power(img, 1/2.4) - 0.055
    )

    img = (img * 255).astype(np.uint8)

    iio.imwrite(filename, img) 

def optimize(
    device: spy.Device,
    reference: spy.Texture,
    stage: Stage,
    post_processor: PostProcessor,
    output_dir: pathlib.Path,
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

    raw_primal = device.create_texture(
        format=spy.Format.rgba32_float,
        width=width,
        height=height,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
        label="primal"
    )

    primal = device.create_texture(
        format=spy.Format.rgba32_float,
        width=width,
        height=height,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
        label="post_processed_primal"
    )

    loss_gradient = device.create_texture(
        format=spy.Format.rgba32_float,
        width=width,
        height=height,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
        label="loss_gradient",
    )

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

    n = 25
    losses = np.empty(shape=(n,), dtype=np.float32)
    
    for iteration in tqdm(range(n)):
        renderer.render(
            scene,
            render_target=raw_primal,
            sample_count=1 << 10,
        )

        command_encoder = device.create_command_encoder()
        post_processor.apply(command_encoder, input=raw_primal, output=primal)
        loss.backwards(command_encoder, primal, loss_gradient)
        post_processor.backwards(command_encoder, raw_primal, loss_gradient, adjoint)
        scene.zero_grad(command_encoder)
        device.submit_command_buffer(command_encoder.finish())
        
        losses[iteration] = loss.loss(primal)
        
        backpropagater.backpropagate(
            scene,
            adjoint,
            sample_count=1 << 10,
            error_target=error_target,
        )
        save_img(error_target.to_numpy()[:,:,0:3], output_dir / f"error.png")

        command_encoder = device.create_command_encoder()
        optimizer.step(command_encoder)
        device.submit_command_buffer(command_encoder.finish())
        
        if iteration % 5 == 0:
            save_img(raw_primal.to_numpy(), output_dir / f"primal_raw_{iteration:02}.png")
            save_img(primal.to_numpy(), output_dir / f"primal_{iteration:02}.png")
            if scene.variables.shape.num_parameters > 0:
                values = scene.variables.parameter_buffer.to_numpy().view(np.float32)
                tqdm.write(f"parameters: {values}")

    with open("output/loss_over_iteration.npy", "wb") as f:
        np.save(f, losses)
    scene.download()

def loss_over_roughness(
    device: spy.Device,
    reference: spy.Texture,
    stage: Stage,
    variable: VariableId,
    values: npt.NDArray,
    post_processor: PostProcessor,
    output_dir: pathlib.Path,
): 
    width = reference.width
    height = reference.height

    shader_table_builder = ShaderTableBuilder()
    scene = Scene(device, stage, shader_table_builder)
    path_tracer = PathTracer(device, shader_table_builder)
    backpropagater = ReplayBackpropagater(device, shader_table_builder)

    loss = L2Loss(device, reference)

    raw_primal = device.create_texture(
        format=spy.Format.rgba32_float,
        width=width,
        height=height,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
        label="raw_primal",
    )

    primal = device.create_texture(
        format=spy.Format.rgba32_float,
        width=width,
        height=height,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
        label="primal",
    )

    loss_gradient = device.create_texture(
        format=spy.Format.rgba32_float,
        width=width,
        height=height,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
        label="loss_gradient",
    )

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

    n = values.shape[0]

    losses = np.empty(shape=n, dtype=np.float32)
    gradients = np.empty(shape=n, dtype=np.float32)

    for i in tqdm(range(n)):
        
        variables = scene.variables.parameter_buffer.to_numpy().view(np.float32)
        variables[variable.index] = values[i]
        scene.variables.parameter_buffer.copy_from_numpy(variables)
    
        path_tracer.render(
            scene,
            raw_primal,
            sample_count=1 << 10,
        )

        losses[i] = loss.loss(primal)

        command_encoder = device.create_command_encoder()
        post_processor.apply(command_encoder, raw_primal, primal)
        loss.backwards(command_encoder, primal, loss_gradient)
        post_processor.backwards(command_encoder, raw_primal, loss_gradient, adjoint)
        scene.zero_grad(command_encoder)
        device.submit_command_buffer(command_encoder.finish())
        
        backpropagater.backpropagate(
            scene,
            adjoint,
            sample_count=1 << 10,
            error_target=error_target,
        )
        save_img(error_target.to_numpy()[:,:,0:3], output_dir / f"error.png")

        scene_gradient = scene.gradient.parameter_buffer.to_numpy().view(np.float32)
        gradients[i] = scene_gradient[variable.index]

    with open(output_dir / "loss_over_roughness.npz", "wb") as f:
        np.savez(f, values=values, losses=losses, gradients=gradients)


# def bsdf_scatter(device: spy.Device):
#     module = spy.Module.load_from_file(device, "shaders/bsdf.slang")

#     n = 1 << 9
#     theta = 1.6
#     phi = -np.pi / 4.0

#     wo = spy.float3(np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta))
#     wi = spy.NDBuffer(device, shape=(n,), dtype=spy.float3)
#     pdf = np.empty(shape=(n,), dtype=np.float32)

#     rng = np.random.default_rng(seed=12345)
#     rand_state = rng.integers(0xffffffff, endpoint=True, size=(n,), dtype=np.uint32)
    
#     bsdf_buffer: spy.NDBuffer = module.sample_microfacet_dielectric_ss(
#         wo,
#         wi,
#         pdf,
#         rand_state,
#         1.5,
#         0.4,
#     )

#     wi = wi.to_numpy()
#     bsdf = bsdf_buffer.to_numpy()

#     mask = (wi[:,0] != 0.0) | (wi[:,1] != 0.0) | (wi[:,2] != 0.0)
#     wi = wi[mask,:]
#     pdf = pdf[mask]
#     bsdf = bsdf[mask]

#     bsdf_eval_buffer: spy.NDBuffer = module.evaluate_microfacet_dielectric_ss(
#         wo,
#         wi,
#         1.5,
#         0.4,
#     )

#     bsdf_eval = bsdf_eval_buffer.to_numpy()

#     error = (bsdf - bsdf_eval) / bsdf

#     print(np.median(error[~np.isnan(error)]))

#     mean_bsdf = np.mean(bsdf, axis=1)
#     weight = mean_bsdf / pdf

#     print(wi.shape)
#     print(np.max(weight))

#     fig = plt.figure()
#     ax = fig.add_subplot(projection='3d')
#     ax.set_xlim(-1.0, 1.0)
#     ax.set_ylim(-1.0, 1.0)
#     ax.set_zlim(-1.0, 1.0)
#     ax.set_xlabel("$t$")
#     ax.set_ylabel("$b$")
#     ax.set_zlabel("$n$")
    
#     ax.plot([0.0, wo.x], [0.0, wo.y], [0.0, wo.z], color='black')
#     ax.scatter(wi[:,0], wi[:,1], cast(Any, wi[:,2]), c=weight/np.max(weight), cmap='jet')

#     fig.savefig("output/bsdf_samples.pdf")

def add_xyzrgb_dragon_scene(stage: Stage, width: int, height: int):
    stage.load_gltf("./assets/XYZRGBDragon.glb")

    environment_image = iio.imread("assets/kloppenheim_06_puresky_4k.hdr")
    stage.environment = stage.add_texture(Texture(environment_image))

def add_lantern_scene(stage: Stage, width: int, height: int):
    stage.load_gltf("./assets/Lantern.glb")

    environment_image = iio.imread("assets/kloppenheim_06_puresky_4k.hdr")
    stage.environment = stage.add_texture(Texture(environment_image))

def textured_sphere_scene(device: spy.Device):
    width = 960
    height = 540

    stage = Stage()

    texture_arr = np.zeros(shape=(256, 256, 3), dtype=np.float32)

    for i in range(8):
        for j in range(8):
            texture_arr[32*i:32*(i+1),32*j:32*(j+1),:] = [0.5, 0.5, 0.5] if (i + j) % 2 == 0 else [1.0, 1.0, 1.0]

    transform = Transform.identity()
    transform.rotate_x(-np.pi/2.0)

    stage.add_instance(Instance(
        mesh_id=stage.add_mesh(Mesh.sphere_uv(n=64)),
        material_id=stage.add_material(Material.lambertian(
            reflectance=stage.add_texture(Texture(texture_arr)),
        )),
        transform=transform,
    ))
    stage.camera = PerspectiveCamera(Transform.from_xyz(0.0, 0.0, 5.0), aspect_ratio=width / height)

    environment_image = np.zeros((1, 1, 3), dtype=np.float32)
    stage.environment = stage.add_texture(Texture(environment_image))

    stage.point_light = PointLight(
        position=spy.float3(1.0, 1.0, 1.0),
        intensity=spy.float3(1.0),
    )

    post_processor = PostProcessor(
        device,
        stages=[
            Tonemapper(),
            SrgbEncoder(gamma=2.4),
        ]
    )

    shader_table_builder = ShaderTableBuilder()
    scene = Scene(device, stage, shader_table_builder)

    raw_reference = device.create_texture(
        type=spy.TextureType.texture_2d,
        format=spy.Format.rgba32_float,
        width=width,
        height=height,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
        label="raw_reference",
    )

    reference = device.create_texture(
        type=spy.TextureType.texture_2d,
        format=spy.Format.rgba32_float,
        width=width,
        height=height,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
        label="reference",
    )

    renderer = PathTracer(
        device,
        shader_table_builder,
    )
    renderer.render(
        scene,
        raw_reference,
        sample_count=1<<10,
        seed=1234,
    )

    command_encoder = device.create_command_encoder()
    post_processor.apply(command_encoder, raw_reference, reference)
    device.submit_command_buffer(command_encoder.finish())

    save_img(raw_reference.to_numpy(), pathlib.Path("./output/textured_sphere/raw_reference.png"))
    save_img(reference.to_numpy(), pathlib.Path("./output/textured_sphere/reference.png"))

def hco_bust_scene(device: spy.Device):
    reference_photo = iio.imread("./assets/PhotoRender_data/hco_reference.png").astype(np.float32) / 255.0
    reference_photo = (reference_photo**2.2) / 2.0

    width = reference_photo.shape[1]
    height = reference_photo.shape[0]

    reference_alpha = np.full(shape=(height, width, 1), fill_value=1.0, dtype=np.float32)
    reference_photo = np.concatenate([reference_photo, reference_alpha], axis=2)

    reference = device.create_texture(
        type=spy.TextureType.texture_2d,
        format=spy.Format.rgba32_float,
        width=width,
        height=height,
        usage=spy.TextureUsage.shader_resource,
        data=reference_photo,
    )

    save_img(reference.to_numpy(), pathlib.Path("./output/hco_bust/reference.png"))

    stage = Stage()

    stage.load_gltf("./assets/PhotoRender_data/hc_orsted.glb")
    bust_id = InstanceId(0)
    ground_id = stage.load_obj("./assets/PhotoRender_data/hco_ground.obj")

    roughness = stage.add_variable(Variable(0.1, range=(0.0, 1.0)))

    stage.replace_material(
        stage.get_instance(bust_id).material_id,
        Metals.aluminium(roughness=roughness),
    )
    
    stage.replace_material(
        stage.get_instance(ground_id).material_id,
        Material.lambertian(reflectance=(0.95, 0.95, 0.84)),
    )

    stage.get_instance(ground_id).transform.rotate_x(np.pi / 2.0)

    camera_position = spy.float3(0.28255452843554596, -1.4590566335764603, 0.15820198110093153)
    camera_rotation = spy.float3x3([
        0.9999265930643305, 0.0019213613603717853, 0.011963145626623345,
        0.012015137318730981, -0.029819251871890606, -0.9994830907489195,
        -0.001563636138289551, 0.999553460595442, -0.02984014835257065,
    ])
    camera_inv_matrix = spy.float3x3([
        9.179619514177686e-05, 0.0, -0.2318790936563762,
        0.0, 9.15419203472016e-05, -0.18786894023818881,
        0.0, 0.0, 1.0
    ])
    camera_resolution = spy.uint2(4928, 3264)

    stage.camera = MatrixCamera(
        inv_intrinsic_matrix=camera_inv_matrix,
        resolution=camera_resolution,
        transform=Transform(
            translation=camera_position,
            rotation=spy.math.quat_from_matrix(camera_rotation),
            scale=spy.float3(1.0),
        ),
    )

    environment_image = np.full(shape=(1, 1, 3), fill_value=0.001137, dtype=np.float32)
    # environment_image = np.full(shape=(1, 1, 3), fill_value=0.2, dtype=np.float32)
    stage.environment = stage.add_texture(Texture(environment_image))

    stage.point_light = PointLight(
        position=spy.float3(-0.15290257842979774, 0.0010295320183712445, 0.827301670159169),
        intensity=spy.float3(0.28),    
    )
    # stage.point_light = PointLight(
    #     position=spy.float3(-0.15290257842979774, 0.0010295320183712445, 0.827301670159169),
    #     intensity=spy.float3(0.0),    
    # )

    post_processor = PostProcessor(
        device,
        stages=[
            # Exposure(stops=1.0),
            # SrgbEncoder(gamma=2.2),
        ],
    )

    random.seed(1234)

    loss_over_roughness(
        device,
        reference,
        stage,
        roughness,
        np.linspace(0.45, 1.0, num=100, endpoint=True, dtype=np.float32),
        # np.array([0.5], dtype=np.float32),
        post_processor,
        pathlib.Path("output/hco_bust"),
    )

    # optimize(
    #     device,
    #     reference,
    #     stage,
    #     post_processor,
    #     pathlib.Path("output/hco_bust")
    # )

def main():

    device = spy.create_device(
        include_paths=[pathlib.Path("./shaders")],
        enable_debug_layers=True,
        enable_print=True,
        type=spy.DeviceType.vulkan,
    )

    hco_bust_scene(device)
    # textured_sphere_scene(device)


if __name__ == "__main__":
    main()
