implementing bsdf;

[Differentiable]
public float3 evaluate_lambertian(
    no_diff float3 wo,
    no_diff float3 wi,
    float reflectance_r,
    float reflectance_g,
    float reflectance_b,
)
{
    float3 reflectance = float3(reflectance_r, reflectance_g, reflectance_b);

    if (wi.z * wo.z < 0.0) {
        return float3(0.0);
    }

    return abs(wi.z) * reflectance / pi;
}

public BsdfSample sample_lambertian(
    float3 wo,
    inout uint rand_state,
    float reflectance_r,
    float reflectance_g,
    float reflectance_b,
) {
    float3 reflectance = float3(reflectance_r, reflectance_g, reflectance_b);
    float3 wi = sign(wo.z) * sample_cosine_hemisphere(rand_state);

    BsdfSample sample;
    sample.wi = wi;
    sample.value = abs(wi.z) * reflectance / pi;
    sample.pdf = abs(wi.z) / pi;

    return sample;
}

// [shader("callable")]
// void lambertian(inout MaterialCallData data) {
//     float reflectance_r = 0.5;
//     float reflectance_g = 0.5;
//     float reflectance_b = 0.5;

//     if ((data.flags & MaterialCallData.EVALUATE_FLAG) != 0) {
//         data.evaluation.value = evaluate_lambertian(
//             data.wo,
//             data.evaluation.wi,
//             reflectance_r,
//             reflectance_g,
//             reflectance_b,
//         );
//     }

//     if ((data.flags & MaterialCallData.SAMPLE_FLAG) != 0) {
//         data.sample = sample_lambertian(
//             data.wo,
//             data.rand_state,
//             reflectance_r,
//             reflectance_g,
//             reflectance_b,
//         );
//     }
// }

// [shader("callable")]
// void backpropagate_lambertian(inout BsdfBackpropagation data) {

//     DifferentialPair<float> reflectance_r = diffPair(scene.variables[data.variables_start_index + 0]);
//     DifferentialPair<float> reflectance_g = diffPair(scene.variables[data.variables_start_index + 1]);
//     DifferentialPair<float> reflectance_b = diffPair(scene.variables[data.variables_start_index + 2]);

//     bwd_diff(evaluate_lambertian)(
//         data.wo,
//         data.wi,
//         reflectance_r,
//         reflectance_g,
//         reflectance_b,
//         data.weight,
//     );

//     if (isfinite(reflectance_r.d)) {
//         float reflectance_r_wave = WaveActiveSum(reflectance_r.d);

//         if (WaveIsFirstLane()) {
//             scene.gradient[data.variables_start_index + 0].add(reflectance_r_wave);
//         }
//     }

//     if (isfinite(reflectance_g.d)) {
//         float reflectance_g_wave = WaveActiveSum(reflectance_g.d);

//         if (WaveIsFirstLane()) {
//             scene.gradient[data.variables_start_index + 1].add(reflectance_g_wave);
//         }
//     }

//     if (isfinite(reflectance_b.d)) {
//         float reflectance_b_wave = WaveActiveSum(reflectance_b.d);

//         if (WaveIsFirstLane()) {
//             scene.gradient[data.variables_start_index + 2].add(reflectance_b_wave);
//         }
//     }
// }
