static const float pi = float.getPi();
static const float tau = 2.0 * float.getPi();

static const int max_depth = 8;

RaytracingAccelerationStructure tlas;
RWTexture2D<float4> render_texture;

Texture2D<float3> environment_map;
SamplerState environment_sampler;

struct Path {
    float3 position;
    uint rand_state;
    float3 direction;

    float3 radiance;
    float3 throughput;
    int depth;

    [mutating]
    void terminate() {
        depth = max_depth;
    }

    bool is_terminated() {
        return depth >= max_depth;
    }
}

[shader("miss")]
func miss(inout Path path)
{
    float phi = atan2(path.direction.z, path.direction.x) + pi;
    float theta = acos(path.direction.y);

    float2 uv = float2(phi / tau, theta / pi);

    path.radiance += path.throughput * environment_map.Sample(environment_sampler, uv);
    path.terminate();
}

[shader("closesthit")]
func closest_hit_triangle_primary(inout Path path, BuiltInTriangleIntersectionAttributes attribs)
{
    int geometry_index = GeometryIndex();
    uint primitive_index = PrimitiveIndex();

    path.throughput *= float3(attribs.barycentrics, 0);
    path.depth += 1;

    float t = RayTCurrent();
    float3 p = path.position + t * path.direction;

    float3 wo = -path.direction;
    float3 wi = -sample_cosine_hemisphere(path.rand_state);

    path.position = p;
    path.direction = wi;
}

[shader("raygeneration")]
func ray_gen()
{
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 dim = DispatchRaysDimensions().xy;

    float2 uv = (float2(pixel) + 0.5) / float2(dim);

    Path path = {};

    path.rand_state = tea(0, pixel.y * dim.x + pixel.x);

    path.position = float3(0.0, 0.0, 3.0);
    path.direction = normalize(float3(
        2.0 * uv.x - 1.0,
        1.0 - 2.0 * uv.y,
        -1.0,
    ));

    path.radiance = float3(0.0);
    path.throughput = float3(1.0);

    RayDesc ray;

    for (int i = 0; i < max_depth; i++) {
        if (path.is_terminated()) {
            break;
        }

        ray.Origin = path.position;
        ray.Direction = path.direction;
        ray.TMin = 1e-3;
        ray.TMax = 1e10;

        TraceRay(
            tlas,
            0,
            0xff,
            0 /* RayContributionToHitGroupIndex */,
            0 /* MultiplierForGeometryContributionHitGroupIndex */,
            0 /* MissShaderIndex */,
            ray,
            path,
        );
    }


    render_texture[pixel] = float4(path.radiance, 1.f);
}

// PRNG xorshift seed generator by NVIDIA
uint tea(uint v0, uint v1)
{
    const uint N = 16u; // User specified number of iterations
    uint s0 = 0u;
    for (uint n = 0u; n < N; n++) {
        s0 += 0x9e3779b9u;
        v0 += ((v1 << 4u) + 0xa341316cu) ^ (v1 + s0) ^ ((v1 >> 5u) + 0xc8013ea4u);
        v1 += ((v0 << 4u) + 0xad90777du) ^ (v0 + s0) ^ ((v0 >> 5u) + 0x7e95761eu);
    }

    return v0;
}

// Generate random unsigned int in [0, 2^31-1)
void mcg31(inout uint rand_state) {
    static const uint LCG_A = 1977654935u; // Multiplier from Hui-Ching Tang [EJOR 2007]
    rand_state = (LCG_A * rand_state) & 0x7FFFFFFEu;
}

// Generate random float in [0, 1)
float rnd(inout uint rand_state) {
    mcg31(rand_state);
    return float(rand_state) / float(0x80000000u);
}

float3 sample_cosine_hemisphere(inout uint rand_state) {
    float cos_theta = sqrt(rnd(rand_state));
    float sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    float phi = 2.0 * pi * rnd(rand_state);

    return float3(
        sin_theta * cos(phi),
        sin_theta * sin(phi),
        cos_theta
    );
}
