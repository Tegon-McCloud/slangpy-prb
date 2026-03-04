import util;
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

static const uint max_depth = 8;

const uint sample_seed;
const uint sample_index;

RWTexture2D<float4> render_target;
ParameterBlock<Scene> scene;

struct Ray {
    float3 origin;
    float3 direction;
}

struct HitDesc {
    float distance;
    float3x3 tangent_to_world;
    uint material_index;
}

struct MissDesc {
    float3 radiance;
}

struct RayTraceResponse {
    bool isHit;
    HitDesc hit;
    MissDesc miss;
}

struct Path {
    float3 position;
    uint rand_state;
    float3 direction;

    float3 radiance;
    float3 throughput;
    int depth = 0;
}

[shader("miss")]
void miss_primary(inout RayTraceResponse response)
{
    float3 w = WorldRayDirection();

    float phi = atan2(w.z, w.x) + pi;
    float theta = acos(w.y);

    float2 uv = float2(phi / tau, theta / pi);

    response.isHit = false;
    response.miss.radiance = scene.environment_map.Sample(scene.environment_sampler, uv);
}

[shader("miss")]
void miss_occlusion()
{}

[shader("closesthit")]
void closest_hit_triangle_primary(inout RayTraceResponse response, BuiltInTriangleIntersectionAttributes attribs)
{
    HitDesc hit;

    uint instance_index = InstanceIndex();
    InstanceDesc instance_desc = scene.instance_descs[instance_index];
    MeshDesc mesh_desc = scene.mesh_descs[instance_desc.mesh_index];

    uint primitive_index = PrimitiveIndex();

    uint i0 = scene.indices[mesh_desc.index_offset + primitive_index * 3 + 0];
    uint i1 = scene.indices[mesh_desc.index_offset + primitive_index * 3 + 1];
    uint i2 = scene.indices[mesh_desc.index_offset + primitive_index * 3 + 2];

    float3 n0 = scene.normals[mesh_desc.vertex_offset + i0];
    float3 n1 = scene.normals[mesh_desc.vertex_offset + i1];
    float3 n2 = scene.normals[mesh_desc.vertex_offset + i2];

    float3 barycentrics = float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y, attribs.barycentrics.x, attribs.barycentrics.y);

    float3 n_object = normalize(barycentrics.x * n0 + barycentrics.y * n1 + barycentrics.z * n2);
    float3 n = mul(ObjectToWorld(), float4(n_object, 0.0));

    hit.distance = RayTCurrent();
    hit.tangent_to_world = make_tbn_matrix(n);
    hit.material_index = instance_desc.material_index;

    response.isHit = true;
    response.hit = hit;
}

[shader("anyhit")]
void any_hit_triangle_occlusion(inout bool occluded) {
    occluded = true;
}

float3 sample_bsdf(
    MaterialDesc material_desc,
    float3 wi,
    out float3 weight,
    inout uint rand_state,
    out uint error_state,
) {
    BsdfSample bsdf_sample;
    bsdf_sample.set_parameter_ptr(material_desc.parameters);
    bsdf_sample.rand_state = rand_state;
    bsdf_sample.error_state = 0;
    bsdf_sample.wi = wi;           // reciprocity
    bsdf_sample.ior_outside = 1.0; // TODO: keep track of ior's with stack

    CallShader(material_desc.sample_call_index, bsdf_sample);

    weight = bsdf_sample.weight;
    rand_state = bsdf_sample.rand_state;
    error_state = bsdf_sample.error_state;
    
    return bsdf_sample.wo;
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
        ray.Origin = path.position;
        ray.Direction = path.direction;
        ray.TMin = 1e-3;
        ray.TMax = 1e10;

        RayTraceResponse response;

        TraceRay(
            scene.tlas,
            0,
            0xff,
            0, /* RayContributionToHitGroupIndex */
            0, /* MultiplierForGeometryContributionHitGroupIndex */
            0, /* MissShaderIndex */
            ray,
            response,
        );

        if (!response.isHit) {
            path.radiance = path.throughput * response.miss.radiance;
            break;
        }

        HitDesc hit = response.hit;
        MaterialDesc material_desc = scene.material_descs[hit.material_index];
        
        float3x3 world_to_tangent = transpose(hit.tangent_to_world);
        float3 wo = mul(world_to_tangent, -path.direction);

        float3 bsdf_weight;
        uint error_state;

        float3 wi = sample_bsdf(
            material_desc,
            wo,
            bsdf_weight,
            path.rand_state,
            error_state,
        );

        path.throughput *= bsdf_weight;
        path.depth += 1;

        path.position = path.position + hit.distance * path.direction;
        path.direction = mul(hit.tangent_to_world, wi);
    }

    float3 radiance = render_target[pixel].rgb;
    radiance = lerp(radiance, path.radiance, 1.0 / (sample_index + 1.0));
    render_target[pixel] = float4(radiance, 1.0);
}
