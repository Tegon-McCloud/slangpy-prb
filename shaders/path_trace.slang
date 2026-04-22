import util;
import scene;
import bsdf;

static const uint max_depth = 8;

const uint sample_seed;
const uint sample_index;

RWTexture2D<float4> render_target;

BsdfSample sample_dumb(
    MaterialDesc material_desc,
    float3 wo,
    inout uint rand_state,
) {
    BsdfSample sample;

    sample.wi = sample_sphere(rand_state);
    sample.pdf = 1.0 / (4.0 * pi);

    MaterialCallData material_call_data;

    material_call_data.wo = wo;
    material_call_data.ior_outside = 1.0;
    material_call_data.constants_start_index = material_desc.constants_start_index;
    material_call_data.rand_state = rand_state;
    material_call_data.flags = MaterialCallData.EVALUATE_FLAG;
    material_call_data.evaluation.wi = sample.wi;

    CallShader(material_desc.call_index, material_call_data);

    rand_state = material_call_data.rand_state;
    sample.value = material_call_data.evaluation.value;

    return sample;
}

float3 sample_path(Ray ray, inout uint rand_state) {

    float3 radiance = float3(0.0);
    float3 throughput = float3(1.0);

    for (int i = 0; i < max_depth; i++) {

        RayTraceResponse response = scene.trace_ray(ray, 1e-3, 1e10);
        
        if (!response.isHit) {
            radiance += throughput * response.miss.radiance;
            break;
        }

        HitDesc hit = response.hit;
        float3 position = ray.origin + hit.distance * ray.direction;
        MaterialDesc material_desc = scene.materials.descs[hit.material_index];

        float3x3 world_to_tangent = transpose(hit.tangent_to_world);
        float3 wo = mul(world_to_tangent, -ray.direction);

        LightSample light_sample = scene.sample_light(position, rand_state);

        MaterialCallData material_call_data;

        material_call_data.wo = wo;
        material_call_data.uv = hit.uv;
        material_call_data.ior_outside = 1.0;
        material_call_data.constants_start_index = material_desc.constants_start_index;
        material_call_data.rand_state = rand_state;
        material_call_data.flags = MaterialCallData.EVALUATE_FLAG | MaterialCallData.SAMPLE_FLAG; 

        material_call_data.evaluation.wi = mul(world_to_tangent, light_sample.wi_world);

        CallShader(material_desc.call_index, material_call_data);
        rand_state = material_call_data.rand_state;

        if (any(material_call_data.evaluation.value != float3(0.0))) {
            bool occluded = scene.trace_occlusion_ray(
                Ray(position, light_sample.wi_world),
                1e-3,
                light_sample.distance,
            );

            if (!occluded) {
                radiance += throughput * material_call_data.evaluation.value * light_sample.radiance / light_sample.pdf;
            }
        }

        BsdfSample sample = material_call_data.sample;

        if (all(sample.wi == float3(0.0))) {
            break;
        }

        throughput *= sample.value / sample.pdf;

        ray.origin = position;
        ray.direction = mul(hit.tangent_to_world, sample.wi);
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
