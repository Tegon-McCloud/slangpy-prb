implementing bsdf;

public struct LambertianParameters {
    public float3 color;
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
    
    s.wo = sign(s.wi.z) * sample_cosine_hemisphere(s.rand_state);
    s.weight = parameters.color;
}