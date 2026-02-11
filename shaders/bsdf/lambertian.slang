implementing bsdf;

struct LambertianParameters {
    float3 color;
}

[shader("callable")]
void call_evaluate_lambertian(inout BrdfEvaluation e)
{
    LambertianParameters parameters = e.get_parameters<LambertianParameters>();

    if (dot(e.wi, e.wo) < 0.0) {
        e.value = 0.0;
        return;
    }
    
    e.value = abs(e.wi.z) * parameters.color / pi;
}

[shader("callable")]
void call_sample_lambertian(inout BrdfEvaluation e)
{
    LambertianParameters parameters = e.get_parameters<LambertianParameters>();

    

}