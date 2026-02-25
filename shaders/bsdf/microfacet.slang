implementing bsdf;

struct MicrofacetParameters {
    float roughness;
}

[shader("callable")]
void call_evaluate_microfacet(inout BsdfEvaluation e)
{
    MicrofacetParameters parameters = e.get_parameters<MicrofacetParameters>();
    
}

[shader("callable")]
void call_sample_microfacet(inout BsdfSample s)
{
    MicrofacetParameters parameters = s.get_parameters<MicrofacetParameters>();

    s.wi = sample_cosine_hemisphere(s.rand_state);
    float cos_theta = s.wi.z;

    if (dot(s.wi, s.wo) < 0.0) {
        s.wi = -s.wi;
    }

    s.value = parameters.roughness / pi;
    s.pdf = cos_theta / pi;
}