module bsdf;
__include bsdf.lambertian;
// __include bsdf.specular_conductor;
// __include bsdf.specular_dielectric;
__include bsdf.microfacet_conductor;
__include bsdf.microfacet_dielectric;

import util;

public struct BsdfEvaluation
{
    public uint constants_start_index;
    public uint variables_start_index;
    public uint rand_state;
    public float3 wo;
    public float3 wi;
    public float3 value;
}

public struct BsdfSample
{
    public uint32_t constants_start_index;
    public uint32_t variables_start_index;
    public uint32_t rand_state;
    public uint32_t error_state;
    public float3 wo;
    public float ior_outside;
    public float3 wi;
    public float3 value;
    public float pdf;
}

public struct BsdfBackpropagation
{
    public uint32_t constants_start_index;
    public uint32_t variables_start_index;
    public float3 wo;
    public float3 wi;
    public float3 weight;
}
