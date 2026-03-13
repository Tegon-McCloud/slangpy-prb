import util;
import scene;
import bsdf;

static const uint max_depth = 8;

const uint sample_seed;
const uint sample_count;

Texture2D<float4> adjoint;

RWTexture2D<float4> error_target;

RayTraceResponse trace_ray(Ray ray, float tmin, float tmax) {
    RayDesc ray_desc;

    ray_desc.Origin = ray.origin;
    ray_desc.Direction = ray.direction;
    ray_desc.TMin = tmin;
    ray_desc.TMax = tmax;

    RayTraceResponse response;

    TraceRay(
        scene.tlas,
        0,
        0xff,
        0, /* RayContributionToHitGroupIndex */
        0, /* MultiplierForGeometryContributionHitGroupIndex */
        0, /* MissShaderIndex */
        ray_desc,
        response,
    );

    return response;
}

float3 sample_bsdf(
    MaterialDesc material_desc,
    float3 wo,
    out float3 value,
    out float pdf,
    inout uint rand_state,
    out uint error_state,
) {
    BsdfSample bsdf_sample;
    bsdf_sample.set_parameter_ptr(material_desc.parameters);
    bsdf_sample.rand_state = rand_state;
    bsdf_sample.error_state = 0;
    bsdf_sample.wo = wo;
    bsdf_sample.ior_outside = 1.0; // TODO: keep track of ior's with stack

    CallShader(material_desc.sample_call_index, bsdf_sample);

    value = bsdf_sample.value;
    pdf = bsdf_sample.pdf;
    rand_state = bsdf_sample.rand_state;
    error_state = bsdf_sample.error_state;

    return bsdf_sample.wi;
}

float3 sample_path(Ray ray, inout uint rand_state, out uint depth) {

    float3 radiance = float3(0.0);
    float3 throughput = float3(1.0);

    for (depth = 0; depth < max_depth; depth++) {
        RayTraceResponse response = trace_ray(ray, 1e-3, 1e10);

        if (!response.isHit) {
            radiance = throughput * response.miss.radiance;
            break;
        }

        HitDesc hit = response.hit;
        MaterialDesc material_desc = scene.material_descs[hit.material_index];

        float3x3 world_to_tangent = transpose(hit.tangent_to_world);
        float3 wo = mul(world_to_tangent, -ray.direction);

        float3 bsdf_value;
        float bsdf_pdf;
        uint error_state;

        float3 wi = sample_bsdf(
            material_desc,
            wo,
            bsdf_value,
            bsdf_pdf,
            rand_state,
            error_state,
        );

        throughput *= bsdf_value / bsdf_pdf;

        ray.origin = ray.origin + hit.distance * ray.direction;
        ray.direction = mul(hit.tangent_to_world, wi);
    }

    return radiance;
}

void sample_path_adjoint(Ray ray, float3 radiance, float3 adjoint_radiance, inout uint rand_state, out uint depth) {

    float3 throughput = float3(1.0);

    for (depth = 0; depth < max_depth; depth++) {
        
        RayTraceResponse response = trace_ray(ray, 1e-3, 1e10);

        if (!response.isHit) {
            break;
        }

        HitDesc hit = response.hit;
        MaterialDesc material_desc = scene.material_descs[hit.material_index];

        float3x3 world_to_tangent = transpose(hit.tangent_to_world);
        float3 wo = mul(world_to_tangent, -ray.direction);

        float3 bsdf_value;
        float bsdf_pdf;
        uint error_state;

        float3 wi = sample_bsdf(
            material_desc,
            wo,
            bsdf_value,
            bsdf_pdf,
            rand_state,
            error_state,
        );

        if (material_desc.requires_grad != 0) {

            BsdfBackpropagation b;
            b.set_parameter_ptr(material_desc.parameters);
            b.set_gradients_ptr(material_desc.gradients);
            b.wo = wo;
            b.wi = wi;
            b.weight = adjoint_radiance * radiance / bsdf_value;
            
            CallShader(material_desc.backpropagate_call_index, b);
        }

        throughput *= bsdf_value / bsdf_pdf;

        ray.origin = ray.origin + hit.distance * ray.direction;
        ray.direction = mul(hit.tangent_to_world, wi);
    }
}

[shader("raygeneration")]
void ray_gen()
{
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 dim = DispatchRaysDimensions().xy;

    uint tid = pixel.y * dim.x + pixel.x;

    uint path_seed = combine_seeds(sample_seed, tid);
    uint rand_state = path_seed;
    
    float2 uv = (float2(pixel) + float2(rnd(rand_state), rnd(rand_state))) / float2(dim);
    Ray camera_ray = scene.camera.getRay(uv);

    uint primary_depth;
    float3 radiance = sample_path(camera_ray, rand_state, primary_depth);
    float3 adjoint_radiance = adjoint[pixel].rgb / sample_count;

    if (any(isnan(radiance)) || !all(isfinite(radiance))) {
        return;
    }

    rand_state = path_seed;

    uv = (float2(pixel) + float2(rnd(rand_state), rnd(rand_state))) / float2(dim);
    camera_ray = scene.camera.getRay(uv);
    
    uint adjoint_depth;
    sample_path_adjoint(camera_ray, radiance, adjoint_radiance, rand_state, adjoint_depth);

    if (primary_depth != adjoint_depth) {
        error_target[pixel] = float4(1.0, 0.0, 1.0, 1.0);
    }
}
