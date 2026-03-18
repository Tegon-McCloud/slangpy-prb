import util;
import scene;
import bsdf;

static const uint max_depth = 8;

struct PathTracer {

    
}

const uint sample_seed;
const uint sample_index;

RWTexture2D<float4> render_target;

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

float3 sample_dumb(
    MaterialDesc material_desc,
    float3 wo,
    out float3 value,
    out float pdf,
    inout uint rand_state,
    out uint error_state,
) {
    float3 wi = sample_sphere(rand_state);
    pdf = 1.0 / (4.0 * pi);

    value = material_desc.evaluate_bsdf(wo, wi, rand_state);
    error_state = 0;

    return wi;
}

float3 sample_path(Ray ray, inout uint rand_state) {

    float3 radiance = float3(0.0);
    float3 throughput = float3(1.0);

    for (int i = 0; i < max_depth; i++) {

        RayTraceResponse response = trace_ray(ray, 1e-3, 1e10);

        if (!response.isHit) {
            radiance = throughput * response.miss.radiance;
            break;
        }

        HitDesc hit = response.hit;
        MaterialDesc material_desc = scene.materials.descs[hit.material_index];

        float3x3 world_to_tangent = transpose(hit.tangent_to_world);
        float3 wo = mul(world_to_tangent, -ray.direction);

        float3 bsdf_value;
        float bsdf_pdf;
        uint error_state;

        float3 wi = material_desc.sample_bsdf(
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

[shader("raygeneration")]
void ray_gen()
{
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 dim = DispatchRaysDimensions().xy;

    uint tid = pixel.y * dim.x + pixel.x;
    uint rand_state = combine_seeds(sample_seed, tid);

    float2 uv = (float2(pixel) + float2(rnd(rand_state), rnd(rand_state))) / float2(dim);
    Ray camera_ray = scene.camera.getRay(uv);

    float3 sample = sample_path(camera_ray, rand_state);

    if (!all(isfinite(sample))) {
        sample = float3(0.0);
    }

    float3 avg = lerp(
        render_target[pixel].rgb,
        sample,
        1.0 / (sample_index + 1.0),
    );

    render_target[pixel] = float4(avg, 1.0);
}
