implementing util;

// return value is unnormalized!
public float3 sample_vndf_std_ggx(float3 w, inout uint rand_state) {
    float phi = 2.0 * pi * rnd(rand_state);

    float z = (1.0 - rnd(rand_state)) * (1.0 + w.z) - w.z;
    float sin_theta = sqrt(max(1.0 - z * z, 0.0));
    float x = sin_theta * cos(phi);
    float y = sin_theta * sin(phi);

    return w + float3(x, y, z);
}

public float3 sample_vndf_ggx(float3 w, float2 alpha, inout uint rand_state) {
    float3 w_std = normalize(float3(w.xy * alpha, w.z));
    float3 wm_std = sample_vndf_std_ggx(w_std, rand_state);

    return normalize(float3(wm_std.xy * alpha, wm_std.z));
}

[Differentiable]
public float ndf_ggx(float3 wm, float2 alpha) {
    float a = (wm.x  * wm.x) / (alpha.x * alpha.x)
        + (wm.y * wm.y) / (alpha.y * alpha.y)
        + wm.z * wm.z;

    return 1.0 / (pi * alpha.x * alpha.y * sq(a));
}

public float vndf_ggx(float3 w, float3 wm, float2 alpha) {
    return masking_ggx(w, alpha) * ndf_ggx(wm, alpha) * max(0.0, dot(w, wm)) / w.z;
}

[Differentiable]
public float lambda_ggx(float3 w, float2 alpha) {
    float a = (sq(alpha.x * w.x) + sq(alpha.y * w.y)) / sq(w.z);
    return 0.5 * (sqrt(1.0 + a) - 1.0);
}

[Differentiable]
public float masking_ggx(float3 w, float2 alpha) {
    return 1.0 / (1.0 + lambda_ggx(w, alpha));
}

[Differentiable]
public float masking_shadowing_ggx(float3 wo, float3 wi, float2 alpha) {
    return 1.0 / (1.0 + lambda_ggx(wo, alpha) + lambda_ggx(wi, alpha));
}
