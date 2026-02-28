implementing bsdf;

struct MicrofacetConductorParameters {
    float3 ior;
    float3 extinction;
    float roughness;
}

[shader("callable")]
void call_evaluate_microfacet_conductor_ss(inout BsdfEvaluation e)
{
    MicrofacetConductorParameters parameters = e.get_parameters<MicrofacetConductorParameters>();
}

[shader("callable")]
void call_sample_microfacet_conductor_ss(inout BsdfSample s)
{
    MicrofacetConductorParameters parameters = s.get_parameters<MicrofacetConductorParameters>();
    float alpha = parameters.roughness * parameters.roughness;
    float3 wi = s.wi;

    float ior_i = s.ior_outside;
    
    float3 wm = sample_vndf_ggx(wi, alpha, s.rand_state);
    float cos_theta_im = dot(wi, wm);

    float3 wo = 2.0 * cos_theta_im * wm - wi;

    float3 reflectance;
    for (int i = 0; i < 3; i++) {
        Complex ior_t = Complex(parameters.ior[i], parameters.extinction[i]);
        Complex ior = ior_i / ior_t;

        reflectance[i] = reflectance_fresnel_complex(cos_theta_im, ior);
    }

    float geometry = masking_ggx(abs(wo.z), alpha);

    s.wo = wo;
    s.weight = reflectance * geometry;
}
