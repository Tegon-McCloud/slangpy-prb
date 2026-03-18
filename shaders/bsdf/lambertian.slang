implementing bsdf;

[Differentiable]
public float3 evaluate_lambertian(
    no_diff float3 wo,
    no_diff float3 wi,
    float color_r,
    float color_g,
    float color_b,
)
{
    float3 color = float3(color_r, color_g, color_b);

    if (wi.z * wo.z < 0.0) {
        return float3(0.0);
    }

    return abs(wi.z) * color / pi;
}

public float3 sample_lambertian(
    float3 wo,
    out float3 wi,
    out float pdf,
    inout uint rand_state,
    float color_r,
    float color_g,
    float color_b,
) {
    float3 color = float3(color_r, color_g, color_b);

    wi = sign(wo.z) * sample_cosine_hemisphere(rand_state);
    pdf = abs(wi.z) / pi;
    return abs(wi.z) * color / pi;
}
