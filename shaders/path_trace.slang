static const float pi = float.getPi();
static const float tau = 2.0 * float.getPi();

static const int max_depth = 8;

const uint sample_seed;
RWTexture2D<float4> sample_texture;

struct MeshDesc {
    uint vertex_offset;
    uint index_offset;
}

struct MaterialDesc {
    uint evaluate_call_index;
    uint sample_call_index;
    Ptr<void, Access.Read, AddressSpace.Device> parameters;
}

struct InstanceDesc {
    uint mesh_index;
    uint material_index;
}

struct PerspectiveCamera {
    float3 forward;
    float3 horizontal;
    float3 vertical;
    float3 position;

    Ray getRay(float2 uv) {
        return Ray(
            position,
            normalize(forward + (2.0 * uv.x - 1.0) * horizontal + (1.0 - 2.0 * uv.y) * vertical),
        );
    }

}

struct Scene {
    RaytracingAccelerationStructure tlas;
    Texture2D<float3> environment_map;
    SamplerState environment_sampler;

    PerspectiveCamera camera;

    StructuredBuffer<float3> vertices;
    StructuredBuffer<uint> indices;
    StructuredBuffer<MeshDesc> mesh_descs;

    // StructuredBuffer<uint8_t> material_parameters;
    StructuredBuffer<MaterialDesc> material_descs;
    
    StructuredBuffer<InstanceDesc> instance_descs;
}

ParameterBlock<Scene> scene;

struct Ray {
    float3 origin;
    float3 direction;
}

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

struct BrdfEvaluation {
    /*
        `parameters` pointer is reinterpret cast to a `uint64_t`, since
        pipeline creation fails when a pointer is passed to a callable
        shader.
    */ 
    uint64_t parameters; 
    float3 wi;
    uint rand_state;
    float3 wo;
    float3 value;
}

struct LambertianParameters {
    float3 color;
}

[shader("callable")]
void call_evaluate_lambertian(inout BrdfEvaluation e) {

    LambertianParameters parameters = *reinterpret<LambertianParameters *>(e.parameters);
    e.value = e.wi.z * parameters.color / pi;
}

[shader("callable")]
void call_sample_lambertian(inout BrdfEvaluation e) {

    LambertianParameters parameters = *reinterpret<LambertianParameters *>(e.parameters);
    e.value = e.wi.z * parameters.color / pi;
}

[shader("miss")]
void miss_primary(inout Path path)
{
    float phi = atan2(path.direction.z, path.direction.x) + pi;
    float theta = acos(path.direction.y);

    float2 uv = float2(phi / tau, theta / pi);

    path.radiance += path.throughput * scene.environment_map.Sample(scene.environment_sampler, uv);
    path.terminate();
}

[shader("miss")]
void miss_occlusion()
{}

[shader("closesthit")]
void closest_hit_triangle_primary(inout Path path, BuiltInTriangleIntersectionAttributes attribs)
{
    uint instance_index = InstanceIndex();
    InstanceDesc instance_desc = scene.instance_descs[instance_index];
    MeshDesc mesh_desc = scene.mesh_descs[instance_desc.mesh_index];
    MaterialDesc material_desc = scene.material_descs[instance_desc.material_index];

    uint primitive_index = PrimitiveIndex();

    uint i0 = scene.indices[mesh_desc.index_offset + primitive_index * 3 + 0];
    uint i1 = scene.indices[mesh_desc.index_offset + primitive_index * 3 + 1];
    uint i2 = scene.indices[mesh_desc.index_offset + primitive_index * 3 + 2];

    float3 v0 = scene.vertices[i0];
    float3 v1 = scene.vertices[i1];
    float3 v2 = scene.vertices[i2];

    float3 barycentrics = float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y, attribs.barycentrics.x, attribs.barycentrics.y);

    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;

    float3 n_object = cross(e1, e2);

    float t = RayTCurrent();
    float3 p = path.position + t * path.direction;

    float3 n = mul(ObjectToWorld(), float4(n_object, 0.0));

    if (dot(n, path.direction) > 0.0) {
        n = -n;
    }

    float3x3 tangent_to_world = make_tbn_matrix(n);
    float3x3 world_to_tangent = transpose(tangent_to_world);

    float3 wo = mul(world_to_tangent, -path.direction);
    float3 wi = sample_cosine_hemisphere(path.rand_state);
    float pdf_inv = pi / wi.z;

    BrdfEvaluation evaluation;
    evaluation.parameters = reinterpret<uint64_t>(material_desc.parameters);
    evaluation.rand_state = path.rand_state;
    evaluation.wi = wi;
    evaluation.wo = wo;

    CallShader(material_desc.evaluate_call_index, evaluation);

    path.rand_state = evaluation.rand_state;
    path.throughput *= evaluation.value * pdf_inv;
    path.depth += 1;

    path.position = p;
    path.direction = mul(tangent_to_world, wi);
}

[shader("anyhit")]
void any_hit_triangle_occlusion(inout bool occluded) {
    occluded = true;
}

[shader("raygeneration")]
void ray_gen()
{
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 dim = DispatchRaysDimensions().xy;

    uint tid = pixel.y * dim.x + pixel.x;

    float2 uv = (float2(pixel) + 0.5) / float2(dim);

    Path path = {};

    path.rand_state = tea(sample_seed, tid);

    Ray camera_ray = scene.camera.getRay(uv);

    path.position = camera_ray.origin;
    path.direction = camera_ray.direction;

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
            scene.tlas,
            0,
            0xff,
            0, /* RayContributionToHitGroupIndex */
            0, /* MultiplierForGeometryContributionHitGroupIndex */
            0, /* MissShaderIndex */
            ray,
            path,
        );
    }

    sample_texture[pixel] = float4(path.radiance, 1.0);
}

float3x3 make_tbn_matrix(float3 normal) {
    float signbit = sign(normal.z + 1.0e-16f);
    float a = -1.0f / (1.0f + abs(normal.z));
    float b = normal.x * normal.y * a;

    return transpose(float3x3(
        float3(1.0f + normal.x * normal.x * a, b, -signbit * normal.x),
        float3(signbit * b, signbit * (1.0f + normal.y * normal.y * a), -normal.y),
        normal,
    ));
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
