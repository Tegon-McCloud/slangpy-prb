implementing util;

[Differentiable]
public float reflectance_fresnel_real(float cos_theta_i, float cos_theta_t, float ior) {
    float a_ii = ior * cos_theta_i;
    float a_tt = cos_theta_t;
    float a_it = ior * cos_theta_t;
    float a_ti = cos_theta_i;

    float r_perp = (a_ii - a_tt) / (a_ii + a_tt);
    float r_para = (a_ti - a_it) / (a_ti + a_it);

    return 0.5 * ((r_perp * r_perp) + (r_para * r_para));
}

[Differentiable]
public float reflectance_fresnel_complex(float cos_theta_i, Complex ior) {

    float sin_theta_i_sq = 1.0 - cos_theta_i * cos_theta_i;
    Complex sin_theta_t_sq = sin_theta_i_sq * (ior * ior);
    Complex cos_theta_t = sqrt(1.0 - sin_theta_t_sq);

    Complex r_perp = 
        (ior * cos_theta_i - cos_theta_t) /
        (ior * cos_theta_i + cos_theta_t);
    Complex r_para =
        (cos_theta_i - ior * cos_theta_t) /
        (cos_theta_i + ior * cos_theta_t);

    return 0.5 * (abs_sq(r_perp) + abs_sq(r_para));
}
