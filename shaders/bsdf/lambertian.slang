implementing bsdf;

struct LambertianParameters {
    float3 color;
}

[shader("callable")]
void call_evaluate_lambertian(inout BsdfEvaluation e)
{
    LambertianParameters parameters = e.get_parameters<LambertianParameters>();

    if (dot(e.wi, e.wo) < 0.0) {
        e.value = 0.0;
        return;
    }
    
    e.value = parameters.color / pi;
}

[shader("callable")]
void call_sample_lambertian(inout BsdfSample s)
{
    LambertianParameters parameters = s.get_parameters<LambertianParameters>();

    s.wi = sample_cosine_hemisphere(s.rand_state);
    float cos_theta = s.wi.z;

    if (dot(s.wi, s.wo) < 0.0) {
        s.wi = -s.wi;
    }

    s.value = parameters.color / pi;
    s.pdf = cos_theta / pi;
}