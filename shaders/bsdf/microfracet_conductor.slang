implementing bsdf;

struct MicrofacetConductorParameters {
    float3 ior;
    float3 extinction;
    float roughness;
}

struct MicrofacetConductorGradients {
    Atomic<float> ior_r;
    Atomic<float> ior_g;
    Atomic<float> ior_b;

    Atomic<float> extinction_r;
    Atomic<float> extinction_g;
    Atomic<float> extinction_b;

    Atomic<float> roughness;
}


[Differentiable]
float3 evaluate_microfacet_conductor_ss(
    no_diff float3 wo,
    no_diff float3 wi,
    float3 ior,
    float3 extinction,
    float roughness,
)
{
    if (wo.z * wi.z < 0.0) {
        return float3(0.0);
    }

    float2 alpha = float2(roughness * roughness);

    float3 wm = normalize(wi + wo);
    if (wm.z < 0.0) {
        wm = -wm;
    }

    float cos_theta_om = dot(wo, wm);
    float3 reflectance;

    for (int i = 0; i < 3; i++) {
        Complex ior_t = Complex(ior[i], extinction[i]);
        Complex ior = 1.0 / ior_t;

        reflectance[i] = reflectance_fresnel_complex(abs(cos_theta_om), ior);
    }

    return reflectance * ndf_ggx(wm, alpha) * masking_shadowing_ggx(wo, wi, alpha) / (4.0 * abs(wo.z));
}

float3 sample_microfacet_conductor_ss(
    float3 wo,
    out float3 wi,
    out float pdf,
    inout uint rand_state,
    float3 ior,
    float3 extinction,
    float roughness,
) {
    float alpha = roughness *roughness;

    float ior_i = 1.0;

    float3 wm = sample_vndf_ggx(wo, alpha, rand_state);
    float cos_theta_om = dot(wo, wm);

    wi = 2.0 * cos_theta_om * wm - wo;

    float3 reflectance;
    for (int i = 0; i < 3; i++) {
        Complex ior_t = Complex(ior[i], extinction[i]);
        Complex ior = ior_i / ior_t;

        reflectance[i] = reflectance_fresnel_complex(cos_theta_om, ior);
    }

    pdf = vndf_ggx(wo, wm, alpha)  / (4.0 * cos_theta_om);
    return reflectance * ndf_ggx(wm, alpha) * masking_shadowing_ggx(wo, wi, alpha) / (4.0 * abs(wo.z));
}

[shader("callable")]
void call_evaluate_microfacet_conductor_ss(inout BsdfEvaluation e)
{
    MicrofacetConductorParameters parameters = e.get_parameters<MicrofacetConductorParameters>();
    e.value = evaluate_microfacet_conductor_ss(e.wo, e.wi, parameters.ior, parameters.extinction, parameters.roughness);
}

[shader("callable")]
void call_sample_microfacet_conductor_ss(inout BsdfSample s)
{
    MicrofacetConductorParameters parameters = s.get_parameters<MicrofacetConductorParameters>();
    s.value = sample_microfacet_conductor_ss(s.wo, s.wi, s.pdf, s.rand_state, parameters.ior, parameters.extinction, parameters.roughness);
}

[shader("callable")]
void call_backpropagate_microfacet_conductor_ss(inout BsdfBackpropagation b)
{
    MicrofacetConductorParameters parameters = b.get_parameters<MicrofacetConductorParameters>();
    Ptr<MicrofacetConductorGradients, Access.ReadWrite, AddressSpace.Device> gradients = b.get_gradients<MicrofacetConductorGradients>();

    DifferentialPair<float3> dp_ior = diffPair(parameters.ior);
    DifferentialPair<float3> dp_extinction = diffPair(parameters.extinction); 
    DifferentialPair<float> dp_roughness = diffPair(parameters.roughness);

    bwd_diff(evaluate_microfacet_conductor_ss)(b.wo, b.wi, dp_ior, dp_extinction, dp_roughness, b.weight);

    if (all(isfinite(dp_ior.d))) {
        float3 ior_d = WaveActiveSum(dp_ior.d);

        if (WaveIsFirstLane()) {
            gradients.ior_r.add(ior_d.r);
            gradients.ior_g.add(ior_d.g);
            gradients.ior_b.add(ior_d.b);
        }
    }

    if (all(isfinite(dp_extinction.d))) {

        float3 extinction_d = WaveActiveSum(dp_extinction.d);

        if (WaveIsFirstLane()) {
            gradients.extinction_r.add(extinction_d.r);
            gradients.extinction_g.add(extinction_d.g);
            gradients.extinction_b.add(extinction_d.b);
        }
    }

    if (isfinite(dp_roughness.d)) {
        float roughness_d = WaveActiveSum(dp_roughness.d);

        if (WaveIsFirstLane()) {
            gradients.roughness.add(roughness_d);
        }
    }
}
