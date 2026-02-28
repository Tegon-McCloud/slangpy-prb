implementing bsdf;

struct SpecularConductorParameters {
    float3 ior;
    float3 extinction;
}

[shader("callable")]
void call_evaluate_specular_conductor(inout BsdfEvaluation e)
{
    e.value = 0.0;
}

[shader("callable")]
void call_sample_specular_conductor(inout BsdfSample s)
{
    SpecularConductorParameters parameters = s.get_parameters<SpecularConductorParameters>();
    float3 wi = s.wi;

    float ior_i = s.ior_outside;
    
    float3 reflectance;

    for (int i = 0; i < 3; i++) {
        Complex ior_t = Complex(parameters.ior[i], parameters.extinction[i]);
        Complex ior = ior_i / ior_t;

        reflectance[i] = reflectance_fresnel_complex(wi.z, ior);
    }
    
    s.wo = float3(-s.wi.x, -s.wi.y, s.wi.z);
    s.weight = reflectance;
}