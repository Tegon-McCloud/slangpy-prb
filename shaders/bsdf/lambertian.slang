implementing bsdf;

public struct LambertianParameters {
    public float3 color;
}

public struct LambertianGradients {
    public Atomic<float> r;
    public Atomic<float> g;
    public Atomic<float> b;
}

[shader("callable")]
void call_evaluate_lambertian(inout BsdfEvaluation e)
{
    LambertianParameters parameters = e.get_parameters<LambertianParameters>();

    if (dot(e.wi, e.wo) < 0.0) {
        e.value = 0.0;
        return;
    }
    
    e.value = abs(e.wi.z) * parameters.color / pi;
}

[shader("callable")]
void call_sample_lambertian(inout BsdfSample s)
{
    LambertianParameters parameters = s.get_parameters<LambertianParameters>();
    
    s.wi = sign(s.wo.z) * sample_cosine_hemisphere(s.rand_state);
    s.value = abs(s.wi.z) * parameters.color / pi;
    s.pdf = abs(s.wi.z) / pi;
}

[shader("callable")]
void call_backpropagate_lambertian(inout BsdfBackpropagation b)
{
    Ptr<LambertianGradients, Access.ReadWrite, AddressSpace.Device> gradients = b.get_gradients<LambertianGradients>();

    float cos_theta_i = abs(b.wi.z);
    float3 color_gradient = b.weight * cos_theta_i / pi;

    if (any(isnan(color_gradient))) {
        return;
    }

    gradients.r.add(color_gradient.r);
    gradients.g.add(color_gradient.g);
    gradients.b.add(color_gradient.b);
}
