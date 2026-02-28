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

public float3 sample_vndf_ggx(float3 w, float alpha, inout uint rand_state) {
    float3 w_std = normalize(float3(w.xy * alpha, w.z));
    float3 wm_std = sample_vndf_std_ggx(w_std, rand_state);

    return normalize(float3(wm_std.xy * alpha, wm_std.z));
}

public float lambda_ggx(float cos_theta, float alpha_sq) {
    float cos_theta_sq = cos_theta * cos_theta;
    float sin_theta_sq = max(1.0 - cos_theta_sq, 0.0);
    float tan_theta_sq = sin_theta_sq / cos_theta_sq;

    return 0.5 * (sign(cos_theta) * sqrt(1.0 + alpha_sq * tan_theta_sq) - 1.0);
}

public float masking_ggx(float cos_theta, float alpha) {
    return 1.0 / (1.0 + lambda_ggx(cos_theta, alpha * alpha));
}
