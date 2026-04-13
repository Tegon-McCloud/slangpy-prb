module bsdf;
__include bsdf.lambertian;
// __include bsdf.specular_conductor;
// __include bsdf.specular_dielectric;
// __include bsdf.microfacet_conductor;
// __include bsdf.microfacet_dielectric;

import util;

public struct BsdfSample
{
    public float3 wi;
    public float3 value;
    public float pdf;
}

public struct BsdfEvaluation
{
    public float3 wi;
    public float3 value;
}

public struct MaterialBackpropagation
{
    public uint32_t constants_start_index;
    public float2 uv;
    public float3 wo;
    public float3 wi;
    public float3 weight;
}

public struct MaterialCallData {
    public static const uint EVALUATE_FLAG = 1 << 0;
    public static const uint SAMPLE_FLAG = 1 << 1;

    public uint32_t constants_start_index;
    public uint32_t rand_state;
    public uint32_t flags;
    public float2 uv;
    public float3 wo;
    public float ior_outside;

    public BsdfEvaluation evaluation;
    public BsdfSample sample;
}
