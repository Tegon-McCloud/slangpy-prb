implementing bsdf;

struct MicrofacetParameters {
    float roughness;
    float ior; 
}


[shader("callable")]
void call_evaluate_microfacet_ss(inout BsdfEvaluation e)
{
    MicrofacetParameters parameters = e.get_parameters<MicrofacetParameters>();
}

[shader("callable")]
void call_sample_microfacet_ss(inout BsdfSample s)
{
    MicrofacetParameters parameters = s.get_parameters<MicrofacetParameters>();
    float alpha = parameters.roughness * parameters.roughness;
    float3 wi = s.wi;
    bool entering = wi.z > 0.0;

    float ior_i = select(entering, s.ior_outside, parameters.ior);
    float ior_t = select(entering, parameters.ior, s.ior_outside);
    float ior = ior_i / ior_t;

    float3 wm;
    if (entering) {
        wm = sample_vndf_ggx(wi, alpha, s.rand_state);
    } else {
        wm = -sample_vndf_ggx(-wi, alpha, s.rand_state);
    }

    float cos_theta_im = dot(wi, wm);
    float sin_theta_im_sq = 1.0 - cos_theta_im * cos_theta_im;
    float sin_theta_tm_sq = ior * ior * sin_theta_im_sq;

    float reflectance = 1.0;
    float cos_theta_tm;

    if (sin_theta_tm_sq < 1.0) {
        cos_theta_tm = sqrt(1.0 - sin_theta_tm_sq);
        reflectance = reflectance_fresnel_real(cos_theta_im, cos_theta_tm, ior);
    }

    float3 wo;

    if (rnd(s.rand_state) < reflectance) {
        wo = 2.0 * cos_theta_im * wm - wi;

        // check that we didn't switch side of the surface during reflection
        if ((wo.z > 0.0) != (wi.z > 0.0)) {
            wo = float3(0.0);
        }
    } else {
        wo = (ior * cos_theta_im - cos_theta_tm) * wm - ior * wi;

        // check that we switched side of the surface during refraction
        if ((wo.z > 0.0) == (wi.z > 0.0)) {
            wo = float3(0.0);
        }
    }

    float geometry = masking_ggx(abs(wo.z), alpha);

    s.wo = wo;
    s.weight = geometry;
}
