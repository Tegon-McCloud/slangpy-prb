module util;

__include util.complex;
__include util.random;
__include util.scattering;
__include util.microfacet;


public static const float pi = float.getPi();
public static const float tau = 2.0 * float.getPi();

public struct Ray {
    public float3 origin;
    public float3 direction;
}

public struct HitDesc {
    public float distance;
    public float3x3 tangent_to_world;
    public uint material_index;
}

public struct MissDesc {
    public float3 radiance;
}

public struct RayTraceResponse {
    public bool isHit;
    public HitDesc hit;
    public MissDesc miss;
}

[ForceInline]
[Differentiable]
public float sq(float x) {
    return x * x;
}

[ForceInline]
[Differentiable]
public vector<float, n> sq<let n : int>(vector<float, n> x) {
    return x * x;
}

public float3x3 make_tbn_matrix(float3 normal) {
    float signbit = sign(normal.z + 1.0e-16f);
    float a = -1.0f / (1.0f + abs(normal.z));
    float b = normal.x * normal.y * a;

    return transpose(float3x3(
        float3(1.0f + normal.x * normal.x * a, b, -signbit * normal.x),
        float3(signbit * b, signbit * (1.0f + normal.y * normal.y * a), -normal.y),
        normal,
    ));
}
