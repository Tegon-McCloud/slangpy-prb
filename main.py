import numpy as np
import slangpy as spy
from PIL import Image

import pathlib


def main():
    device = spy.create_device(
        include_paths=[pathlib.Path("./shaders")],
        enable_debug_layers=True,
        type=spy.DeviceType.vulkan
    )

    vertices = np.array([
        -1.0, 0.0, 0.0,
        0.0, -1.0, 0.0,
        1.0, 1.0, 0.0,    
    ], dtype=np.float32)

    indices = np.array([0, 1, 2], dtype=np.uint32)

    vertex_buffer = device.create_buffer(
        usage=spy.BufferUsage.shader_resource,
        label="vertex_buffer",
        data=vertices,    
    )
    index_buffer = device.create_buffer(
        usage=spy.BufferUsage.shader_resource,
        label="index_buffer",
        data=indices,
    )

    blas_input_triangles = spy.AccelerationStructureBuildInputTriangles({
        "vertex_buffers": [vertex_buffer],
        "vertex_format": spy.Format.rgb32_float,
        "vertex_count": vertices.size // 3,
        "vertex_stride": vertices.itemsize * 3,
        "index_buffer": index_buffer,
        "index_format": spy.IndexFormat.uint32,
        "index_count": index_buffer.size,
        "flags": spy.AccelerationStructureGeometryFlags.opaque,
    })

    blas_build_desc = spy.AccelerationStructureBuildDesc({
        "inputs": [blas_input_triangles],
    })

    blas_sizes = device.get_acceleration_structure_sizes(blas_build_desc)

    blas_scratch_buffer = device.create_buffer(
        size=blas_sizes.scratch_size,
        usage=spy.BufferUsage.unordered_access,
        label="blas_scratch_buffer",
    )

    blas = device.create_acceleration_structure(
        size=blas_sizes.acceleration_structure_size,
        label="blas",
    )

    command_encoder = device.create_command_encoder()
    command_encoder.build_acceleration_structure(
        desc=blas_build_desc,
        dst=blas,
        src=None,
        scratch_buffer=blas_scratch_buffer,
    )
    device.submit_command_buffer(command_encoder.finish())

    instance_list = device.create_acceleration_structure_instance_list(1)
    instance_list.write(
        0,
        {
            "transform": spy.float3x4.identity(),
            "instance_id": 0,
            "instance_mask": 0xff,
            "instance_contribution_to_hit_group_index": 0,
            "flags": spy.AccelerationStructureInstanceFlags.none,
            "acceleration_structure": blas.handle,
        },
    )

    tlas_build_desc = spy.AccelerationStructureBuildDesc({
        "inputs": [instance_list.build_input_instances()]
    })

    tlas_sizes = device.get_acceleration_structure_sizes(tlas_build_desc)

    tlas_scratch_buffer = device.create_buffer(
        size=tlas_sizes.scratch_size,
        usage=spy.BufferUsage.unordered_access,
        label="tlas_scratch_buffer",
    )

    tlas = device.create_acceleration_structure(
        size=tlas_sizes.acceleration_structure_size,
        label="tlas",
    )

    command_encoder = device.create_command_encoder()
    command_encoder.build_acceleration_structure(
        desc=tlas_build_desc,
        dst=tlas,
        src=None,
        scratch_buffer=tlas_scratch_buffer,
    )
    device.submit_command_buffer(command_encoder.finish())

    texture_loader = spy.TextureLoader(device)
    environment_map = texture_loader.load_texture(spy.Bitmap.load_from_file("./assets/kloppenheim_06_puresky_4k.hdr"))

    render_texture = device.create_texture(
        format = spy.Format.rgba32_float,
        width=1024,
        height=1024,
        usage=spy.TextureUsage.unordered_access,
        label="render_texture",
    )

    program = device.load_program("raytracing_example.slang", ["ray_gen", "miss", "closest_hit_triangle_primary"])

    pipeline = device.create_ray_tracing_pipeline(
        program=program,
        hit_groups=[
            spy.HitGroupDesc(hit_group_name="hit_group", closest_hit_entry_point="closest_hit_triangle_primary"),
        ],
        max_recursion=1,
        max_ray_payload_size=48,
    )


    shader_table = device.create_shader_table(
        program=program,
        ray_gen_entry_points=["ray_gen"],
        miss_entry_points=["miss"],
        hit_group_names=["hit_group"],
    )

    command_encoder = device.create_command_encoder()
    with command_encoder.begin_ray_tracing_pass() as pass_encoder:
        shader_object = pass_encoder.bind_pipeline(pipeline, shader_table)
        cursor = spy.ShaderCursor(shader_object)
        cursor.tlas = tlas
        cursor.render_texture = render_texture
        cursor.environment_map = environment_map

        pass_encoder.dispatch_rays(0, [1024, 1024, 1])
    
    device.submit_command_buffer(command_encoder.finish())

    result = render_texture.to_numpy()
    result = np.clip(result, 0.0, 1.0)
    result = (result * 255).astype(np.uint8)

    Image.fromarray(result).save("./output/example.png")     

if __name__ == "__main__":
    main()
