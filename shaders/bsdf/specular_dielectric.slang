implementing bsdf;

struct SpecularDielectricParameters {
    float ior;
    float extinction;
}

[shader("callable")]
void call_evaluate_specular_dielectric(inout BsdfEvaluation e)
{
    e.value = 0.0;
}

[shader("callable")]
void call_sample_specular_dielectric(inout BsdfSample s)
{
    SpecularDielectricParameters parameters = s.get_parameters<SpecularDielectricParameters>();
    float3 wi = s.wi;
    bool entering = wi.z > 0.0;

    float ior_i = select(entering, s.ior_outside, parameters.ior);
    float ior_t = select(entering, parameters.ior, s.ior_outside);
    float ior = ior_i / ior_t;

    float3 wn = float3(0.0, 0.0, select(entering, 1.0, -1.0));

    float cos_theta_i = dot(wi, wn);
    float sin_theta_i_sq = 1.0 - cos_theta_i * cos_theta_i;
    float sin_theta_t_sq = ior * ior * sin_theta_i_sq;

    float reflectance = 1.0;
    float cos_theta_t;

    if (sin_theta_t_sq < 1.0) {
        cos_theta_t = sqrt(1.0 - sin_theta_t_sq);
        reflectance = reflectance_fresnel_real(cos_theta_i, cos_theta_t, ior);
    }

    if (rnd(s.rand_state) < reflectance) {
        s.wo = 2.0 * cos_theta_i * wn - wi;
    } else {
        s.wo = (ior * cos_theta_i - cos_theta_t) * wn - ior * wi;

        // validation
        if (abs(length(s.wo) - 1.0) > 1e-3) {
            s.error_state = 1;
        } 

        if ((s.wo.z > 0.0) == (wi.z > 0.0)) {
            s.error_state = 2;
        }
    }
    s.weight = 1.0;
}