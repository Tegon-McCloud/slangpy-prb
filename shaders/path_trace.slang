import util;
import random;
import bsdf;

struct MeshDesc {
    uint vertex_offset;
    uint index_offset;
}

struct MaterialDesc {
    uint evaluate_call_index;
    uint sample_call_index;
    Ptr<void, Access.Read, AddressSpace.Device> parameters;
}

struct InstanceDesc {
    uint mesh_index;
    uint material_index;
}

struct PerspectiveCamera {
    float3 forward;
    float3 horizontal;
    float3 vertical;
    float3 position;

    Ray getRay(float2 uv) {
        return Ray(
            position,
            normalize(forward + (2.0 * uv.x - 1.0) * horizontal + (1.0 - 2.0 * uv.y) * vertical),
        );
    }

}

struct Scene {
    RaytracingAccelerationStructure tlas;
    Texture2D<float3> environment_map;
    SamplerState environment_sampler;

    PerspectiveCamera camera;

    StructuredBuffer<float3> positions;
    StructuredBuffer<float3> normals;
    StructuredBuffer<uint> indices;
    StructuredBuffer<MeshDesc> mesh_descs;

    StructuredBuffer<MaterialDesc> material_descs;
    
    StructuredBuffer<InstanceDesc> instance_descs;
}

static const int max_depth = 8;

const uint sample_seed;
RWTexture2D<float4> sample_texture;
ParameterBlock<Scene> scene;

struct Ray {
    float3 origin;
    float3 direction;
}

struct Path {
    float3 position;
    uint rand_state;
    float3 direction;

    float3 radiance;
    float3 throughput;
    int depth = 0;

    [mutating]
    void terminate() {
        depth = max_depth;
    }

    bool is_terminated() {
        return depth >= max_depth;
    }
}



[shader("miss")]
void miss_primary(inout Path path)
{
    float phi = atan2(path.direction.z, path.direction.x) + pi;
    float theta = acos(path.direction.y);

    float2 uv = float2(phi / tau, theta / pi);

    path.radiance += path.throughput * scene.environment_map.Sample(scene.environment_sampler, uv);
    path.terminate();
}

[shader("miss")]
void miss_occlusion()
{}

[shader("closesthit")]
void closest_hit_triangle_primary(inout Path path, BuiltInTriangleIntersectionAttributes attribs)
{
    uint instance_index = InstanceIndex();
    InstanceDesc instance_desc = scene.instance_descs[instance_index];
    MeshDesc mesh_desc = scene.mesh_descs[instance_desc.mesh_index];
    MaterialDesc material_desc = scene.material_descs[instance_desc.material_index];

    uint primitive_index = PrimitiveIndex();

    uint i0 = scene.indices[mesh_desc.index_offset + primitive_index * 3 + 0];
    uint i1 = scene.indices[mesh_desc.index_offset + primitive_index * 3 + 1];
    uint i2 = scene.indices[mesh_desc.index_offset + primitive_index * 3 + 2];

    float3 n0 = scene.normals[mesh_desc.vertex_offset + i0];
    float3 n1 = scene.normals[mesh_desc.vertex_offset + i1];
    float3 n2 = scene.normals[mesh_desc.vertex_offset + i2];

    float3 barycentrics = float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y, attribs.barycentrics.x, attribs.barycentrics.y);

    float3 n_object = normalize(barycentrics.x * n0 + barycentrics.y * n1 + barycentrics.z * n2);

    float t = RayTCurrent();
    float3 p = path.position + t * path.direction;

    float3 n = mul(ObjectToWorld(), float4(n_object, 0.0));

    if (dot(n, path.direction) > 0.0) {
        n = -n;
    }

    float3x3 tangent_to_world = make_tbn_matrix(n);
    float3x3 world_to_tangent = transpose(tangent_to_world);
    
    float3 wo = mul(world_to_tangent, -path.direction);
    
    BsdfSample bsdf_sample;
    bsdf_sample.set_parameter_ptr(material_desc.parameters);
    bsdf_sample.wo = wo;
    bsdf_sample.rand_state = path.rand_state;

    CallShader(material_desc.sample_call_index, bsdf_sample);

    float3 wi = bsdf_sample.wi;

    path.rand_state = bsdf_sample.rand_state;
    path.throughput *= bsdf_sample.value / bsdf_sample.pdf;
    path.depth += 1;

    path.position = p;
    path.direction = mul(tangent_to_world, wi);
}

[shader("anyhit")]
void any_hit_triangle_occlusion(inout bool occluded) {
    occluded = true;
}

[shader("raygeneration")]
void ray_gen()
{
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 dim = DispatchRaysDimensions().xy;

    uint tid = pixel.y * dim.x + pixel.x;

    
    Path path = {};
    
    path.rand_state = combine_seeds(sample_seed, tid);
    
    float2 uv = (float2(pixel) + float2(rnd(path.rand_state), rnd(path.rand_state))) / float2(dim);
    Ray camera_ray = scene.camera.getRay(uv);

    path.position = camera_ray.origin;
    path.direction = camera_ray.direction;
    path.radiance = float3(0.0);
    path.throughput = float3(1.0);

    RayDesc ray;

    for (int i = 0; i < max_depth; i++) {
        if (path.is_terminated()) {
            break;
        }

        ray.Origin = path.position;
        ray.Direction = path.direction;
        ray.TMin = 1e-3;
        ray.TMax = 1e10;

        TraceRay(
            scene.tlas,
            0,
            0xff,
            0, /* RayContributionToHitGroupIndex */
            0, /* MultiplierForGeometryContributionHitGroupIndex */
            0, /* MissShaderIndex */
            ray,
            path,
        );
    }

    sample_texture[pixel] = float4(path.radiance, 1.0);
}

[shader("raygen")]
void ray_gen_adjoint() {
    

}

float3x3 make_tbn_matrix(float3 normal) {
    float signbit = sign(normal.z + 1.0e-16f);
    float a = -1.0f / (1.0f + abs(normal.z));
    float b = normal.x * normal.y * a;

    return transpose(float3x3(
        float3(1.0f + normal.x * normal.x * a, b, -signbit * normal.x),
        float3(signbit * b, signbit * (1.0f + normal.y * normal.y * a), -normal.y),
        normal,
    ));
}

