implementing bsdf;

public struct LambertianParameters {
    public float3 color;
}

public struct LambertianGradients {
    public Atomic<float> r;
    public Atomic<float> g;
    public Atomic<float> b;
}

[Differentiable]
float3 evaluate_lambertian(
    no_diff float3 wo,
    no_diff float3 wi,
    float3 color,
)
{
    if (wi.z * wo.z < 0.0) {
        return float3(0.0);
    }

    return abs(wi.z) * color / pi;
}

float3 sample_lambertian(
    float3 wo,
    out float3 wi,
    out float pdf,
    inout uint rand_state,
    float3 color,
) {
    wi = sign(wo.z) * sample_cosine_hemisphere(rand_state);
    pdf = abs(wi.z) / pi;
    return abs(wi.z) * color / pi;
}

[shader("callable")]
void call_evaluate_lambertian(inout BsdfEvaluation e)
{
    LambertianParameters parameters = e.get_parameters<LambertianParameters>();
    e.value = evaluate_lambertian(e.wo, e.wi, parameters.color);
}

[shader("callable")]
void call_sample_lambertian(inout BsdfSample s)
{
    LambertianParameters parameters = s.get_parameters<LambertianParameters>();
    s.value = sample_lambertian(s.wo, s.wi, s.pdf, s.rand_state, parameters.color);
}

[shader("callable")]
void call_backpropagate_lambertian(inout BsdfBackpropagation b)
{
    LambertianParameters parameters = b.get_parameters<LambertianParameters>();
    Ptr<LambertianGradients, Access.ReadWrite, AddressSpace.Device> gradients = b.get_gradients<LambertianGradients>();

    DifferentialPair<float3> dp_color = diffPair(parameters.color);
    
    bwd_diff(evaluate_lambertian)(b.wi, b.wo, dp_color, b.weight);

    if (all(isfinite(dp_color.d))) {

        float3 color_d = WaveActiveSum(dp_color.d);

        if (WaveIsFirstLane()) {
            gradients.r.add(color_d.r);
            gradients.g.add(color_d.g);
            gradients.b.add(color_d.b);
        }
    }


}
