implementing bsdf;

struct MicrofacetDielectricParameters {
    float ior;
    float roughness;
}

struct MicrofacetDielectricGradients {
    Atomic<float> ior;
    Atomic<float> roughness;
}

[Differentiable]
float3 evaluate_microfacet_dielectric_ss(
    no_diff float3 wo,
    no_diff float3 wi,
    no_diff float ior,
    float roughness,
) {
    float2 alpha = float2(roughness * roughness);
    bool reflect = wo.z * wi.z > 0.0;
    bool entering = wo.z > 0.0;

    float ior_o = select(entering, 1.0, ior);
    float ior_t = select(entering, ior, 1.0);
    float ior_i = select(reflect, ior_o, ior_t);
    float ior = ior_o / ior_t;

    float3 wm = normalize(ior_i * wi + ior_o * wo);
    if (wm.z < 0.0) {
        wm = -wm;
    }

    float cos_theta_om = dot(wo, wm);
    float cos_theta_im = dot(wi, wm);

    if (cos_theta_om * wo.z < 0.0 || cos_theta_im * wi.z < 0.0) {
        return float3(0.0);
    }

    float sin_theta_om_sq = 1.0 - cos_theta_om * cos_theta_om;
    float sin_theta_tm_sq = ior * ior * sin_theta_om_sq;

    float reflectance = 1.0;

    if (sin_theta_tm_sq < 1.0) {
        float cos_theta_tm = sqrt(1.0 - sin_theta_tm_sq);
        reflectance = reflectance_fresnel_real(abs(cos_theta_om), cos_theta_tm, ior);
    }

    float cosines;
    float fresnel;

    if (reflect) {
        fresnel = reflectance;
        cosines = 1.0 / (4.0 * abs(wo.z));
    } else {
        fresnel = 1.0 - reflectance;
        float denom = sq(cos_theta_im + ior * cos_theta_om) * abs(wo.z);
        cosines = abs(cos_theta_im) * abs(cos_theta_om) / denom;
    }

    return fresnel * ndf_ggx(wm, alpha) * masking_shadowing_ggx(wo, wi, alpha) * cosines;
}

float3 sample_microfacet_dielectric_ss(
    float3 wo,
    out float3 wi,
    out float pdf,
    inout uint rand_state,
    float ior,
    float roughness,
) {
    float2 alpha = float2(roughness * roughness);
    bool entering = wo.z > 0.0;

    float ior_o = select(entering, 1.0, ior);
    float ior_t = select(entering, ior, 1.0);
    float ior = ior_o / ior_t;

    float3 wm;
    float vndf;
    float ndf;

    if (entering) {
        wm = sample_vndf_ggx(wo, alpha, rand_state);
        vndf = vndf_ggx(wo, wm, alpha);
        ndf = ndf_ggx(wm, alpha);
    } else {
        wm = -sample_vndf_ggx(-wo, alpha, rand_state);
        vndf = vndf_ggx(-wo, -wm, alpha);
        ndf = ndf_ggx(-wm, alpha);
    }

    float cos_theta_om = dot(wo, wm);
    float sin_theta_om_sq = 1.0 - cos_theta_om * cos_theta_om;
    float sin_theta_tm_sq = ior * ior * sin_theta_om_sq;

    float reflectance = 1.0;
    float cos_theta_tm;

    if (sin_theta_tm_sq < 1.0) {
        cos_theta_tm = sqrt(1.0 - sin_theta_tm_sq);
        reflectance = reflectance_fresnel_real(cos_theta_om, cos_theta_tm, ior);
    }

    float fresnel;
    float cosines;

    if (rnd(rand_state) < reflectance) {
        wi = 2.0 * cos_theta_om * wm - wo;
    
        // check that we didn't switch side of the surface during reflection
        if ((wi.z > 0.0) != (wo.z > 0.0)) {
            wi = float3(0.0);
        }

        fresnel = reflectance;
        pdf = fresnel * vndf / (4.0 * cos_theta_om);
        cosines = 1.0 / (4.0 * abs(wo.z));
    } else {

        wi = (ior * cos_theta_om - cos_theta_tm) * wm - ior * wo;
        
        // check that we switched side of the surface during refraction
        if ((wi.z > 0.0) == (wo.z > 0.0)) {
            wi = float3(0.0);
        }
        
        float denom = sq(cos_theta_tm + ior * cos_theta_om);
        float jacobian = cos_theta_tm / denom;
        fresnel = 1.0 - reflectance;
        pdf = fresnel * vndf * jacobian;
        cosines = cos_theta_om / abs(wo.z) * jacobian;
    }

    float geometry = masking_shadowing_ggx(wo, wi, alpha);

    return fresnel * ndf * geometry * cosines;
}


[shader("callable")]
void call_evaluate_microfacet_dielectric_ss(inout BsdfEvaluation e)
{
    MicrofacetDielectricParameters parameters = e.get_parameters<MicrofacetDielectricParameters>();
    e.value = evaluate_microfacet_dielectric_ss(e.wo, e.wi, parameters.ior, parameters.roughness);
}

[shader("callable")]
void call_sample_microfacet_dielectric_ss(inout BsdfSample s)
{
    MicrofacetDielectricParameters parameters = s.get_parameters<MicrofacetDielectricParameters>();
    s.value = sample_microfacet_dielectric_ss(s.wo, s.wi, s.pdf, s.rand_state, parameters.ior, parameters.roughness);
}

[shader("callable")]
void call_backpropagate_microfacet_dielectric_ss(inout BsdfBackpropagation b)
{
    MicrofacetDielectricParameters parameters = b.get_parameters<MicrofacetDielectricParameters>();
    Ptr<MicrofacetDielectricGradients, Access.ReadWrite, AddressSpace.Device> gradients = b.get_gradients<MicrofacetDielectricGradients>();

    DifferentialPair<float> dp_roughness = diffPair(parameters.roughness);

    bwd_diff(evaluate_microfacet_dielectric_ss)(b.wi, b.wo, parameters.ior, dp_roughness, b.weight);

    if (any(isnan(dp_roughness.d))) {
        return;
    }

    gradients.roughness.add(dp_roughness.d);

}
