import util;
import bsdf;

public struct MeshDesc {
    public uint vertex_offset;
    public uint index_offset;
}

public struct MaterialDesc {
    public uint call_index;
    public uint backpropagate_call_index;
    public uint requires_grad;
    public uint constants_start_index;
}

public struct InstanceDesc {
    public uint mesh_index;
    public uint material_index;
}

extern public struct Camera : ICamera;

public struct Meshes {
    public StructuredBuffer<float3> positions;
    public StructuredBuffer<float3> normals;
    public StructuredBuffer<float2> uvs;
    public StructuredBuffer<uint> indices;
    public StructuredBuffer<MeshDesc> descs;
}

public struct Materials {
    public StructuredBuffer<float> constants;
    public StructuredBuffer<MaterialDesc> descs;
}

public struct PointLight {
    public float3 position;
    public float3 intensity;   
}

public struct Lights {
    public PointLight point_light;
    public uint environment_texture_index;
}

public struct LightSample {
    public float3 wi_world;
    public float distance;
    public float3 radiance;
    public float pdf;
}

public struct Scene {
    public RaytracingAccelerationStructure tlas;

    public Camera camera;
    public Meshes meshes;
    public StructuredBuffer<Texture2D.Handle> textures;
    public Materials materials;
    public Lights lights;    

    public StructuredBuffer<InstanceDesc> instance_descs;
    
    public StructuredBuffer<float> variables;
    public RWStructuredBuffer<Atomic<float>> gradient;
    
    public SamplerState default_sampler;
    
    public RayTraceResponse trace_ray(Ray ray, float tmin, float tmax) {
        RayDesc ray_desc;

        ray_desc.Origin = ray.origin;
        ray_desc.Direction = ray.direction;
        ray_desc.TMin = tmin;
        ray_desc.TMax = tmax;

        RayTraceResponse response;

        TraceRay(
            tlas,
            0,
            0xff,
            0, /* RayContributionToHitGroupIndex */
            0, /* MultiplierForGeometryContributionHitGroupIndex */
            0, /* MissShaderIndex */
            ray_desc,
            response,
        );

        return response;
    }

    public bool trace_occlusion_ray(Ray ray, float tmin, float tmax) {

        RayDesc ray_desc;

        ray_desc.Origin = ray.origin;
        ray_desc.Direction = ray.direction;
        ray_desc.TMin = tmin;
        ray_desc.TMax = tmax;

        bool occluded = false;

        TraceRay(
            tlas,
            RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
            0xff,
            1, /* RayContributionToHitGroupIndex */
            0, /* MultiplierForGeometryContributionHitGroupIndex */
            1, /* MissShaderIndex */
            ray_desc,
            occluded,
        );

        return occluded;
    }


    public LightSample sample_light(float3 position, inout uint rand_state) {

        float3 diff = lights.point_light.position - position;
        float dist_sq = dot(diff, diff);
        float3 wi_world = normalize(diff);

        return LightSample(
            wi_world,
            sqrt(dist_sq),
            lights.point_light.intensity * (1.0 / dist_sq),
            1.0,
        );
    }

}

public ParameterBlock<Scene> scene;
