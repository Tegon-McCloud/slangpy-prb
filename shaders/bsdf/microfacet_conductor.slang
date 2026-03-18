implementing bsdf;

[Differentiable]
public float3 evaluate_microfacet_conductor_ss(
    no_diff float3 wo,
    no_diff float3 wi,
    float ior_r,
    float ior_g,
    float ior_b,
    float extinction_r,
    float extinction_g,
    float extinction_b,
    float roughness,
)
{
    float3 ior = float3(ior_r, ior_g, ior_b);
    float3 extinction = float3(extinction_r, extinction_g, extinction_b);
    
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

public float3 sample_microfacet_conductor_ss(
    float3 wo,
    out float3 wi,
    out float pdf,
    inout uint rand_state,
    float ior_r,
    float ior_g,
    float ior_b,
    float extinction_r,
    float extinction_g,
    float extinction_b,
    float roughness,
) {
    float3 ior = float3(ior_r, ior_g, ior_b);
    float3 extinction = float3(extinction_r, extinction_g, extinction_b);
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
