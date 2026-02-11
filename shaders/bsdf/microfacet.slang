implementing bsdf;

struct MicrofacetParameters {
    float roughness;
}

[shader("callable")]
void call_evaluate_microfacet(inout BrdfEvaluation e)
{
    MicrofacetParameters parameters = *reinterpret<MicrofacetParameters *>(e.parameters);

    e.value = e.wi.z * parameters.roughness / pi;
}

[shader("callable")]
void call_sample_microfacet(inout BrdfEvaluation e)
{}
