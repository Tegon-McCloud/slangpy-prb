import util;
import scene;
import bsdf;

static const uint max_depth = 8;

const uint sample_seed;
const uint sample_count;

Texture2D<float4> adjoint;

RWTexture2D<float4> error_target;

float3 sample_path(Ray ray, inout uint rand_state, out uint depth) {
    float3 radiance = float3(0.0);
    float3 throughput = float3(1.0);

    for (depth = 0; depth < max_depth; depth++) {
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

void sample_path_adjoint(Ray ray, float3 radiance, float3 adjoint_radiance, inout uint rand_state, out uint depth) {
    float3 throughput = float3(1.0);

    for (depth = 0; depth < max_depth; depth++) {
        RayTraceResponse response = scene.trace_ray(ray, 1e-3, 1e10);

        if (!response.isHit) {
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
                radiance -= throughput * material_call_data.evaluation.value * light_sample.radiance / light_sample.pdf;

                if (material_desc.requires_grad != 0) {
                    MaterialBackpropagation b;
                    b.constants_start_index = material_desc.constants_start_index;
                    b.uv = hit.uv;
                    b.wo = wo;
                    b.wi = material_call_data.evaluation.wi;
                    b.weight = adjoint_radiance * light_sample.radiance / light_sample.pdf;

                    CallShader(material_desc.backpropagate_call_index, b);
                }
            }
        }

        BsdfSample sample = material_call_data.sample;

        if (all(sample.wi == float3(0.0))) {
            break;
        }

        if (material_desc.requires_grad != 0) {

            MaterialBackpropagation b;
            b.constants_start_index = material_desc.constants_start_index;
            b.uv = hit.uv;
            b.wo = wo;
            b.wi = sample.wi;
            b.weight = adjoint_radiance * radiance / sample.value;

            CallShader(material_desc.backpropagate_call_index, b);
        }

        throughput *= sample.value / sample.pdf;

        ray.origin = position;
        ray.direction = mul(hit.tangent_to_world, sample.wi);
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
    float3 adjoint_test;
    sample_path_adjoint(camera_ray, radiance, adjoint_radiance, rand_state, adjoint_depth);

    if (primary_depth != adjoint_depth) {
        error_target[pixel] = float4(1.0, 0.0, 1.0, 1.0);
    }
}
