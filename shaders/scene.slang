import util;
import bsdf;

public struct MeshDesc {
    public uint vertex_offset;
    public uint index_offset;
}

public struct MaterialDesc {
    public uint evaluate_call_index;
    public uint sample_call_index;
    public uint backpropagate_call_index;
    public uint requires_grad;
    public uint constants_start_index;
    public uint variables_start_index;

    public float3 evaluate_bsdf(
        float3 wo,
        float3 wi,
        inout uint rand_state,
        
    ) {
        BsdfEvaluation e;
        e.constants_start_index = this.constants_start_index;
        e.variables_start_index = this.variables_start_index;
        e.rand_state = rand_state;
        e.wo = wo;
        e.wi = wi;

        CallShader(this.evaluate_call_index, e);

        rand_state = e.rand_state;

        return e.value;
    }

    public float3 sample_bsdf(
        float3 wo,
        out float3 value,
        out float pdf,
        inout uint rand_state,
        out uint error_state,
    ) {
        BsdfSample bsdf_sample;
        bsdf_sample.constants_start_index = this.constants_start_index;
        bsdf_sample.variables_start_index = this.variables_start_index;
        bsdf_sample.rand_state = rand_state;
        bsdf_sample.error_state = 0;
        bsdf_sample.wo = wo;
        bsdf_sample.ior_outside = 1.0; // TODO: keep track of ior

        CallShader(this.sample_call_index, bsdf_sample);

        value = bsdf_sample.value;
        pdf = bsdf_sample.pdf;
        rand_state = bsdf_sample.rand_state;
        error_state = bsdf_sample.error_state;

        return bsdf_sample.wi;
    }
}

public struct InstanceDesc {
    public uint mesh_index;
    public uint material_index;
}

public struct PerspectiveCamera {
    public float3 forward;
    public float3 horizontal;
    public float3 vertical;
    public float3 position;

    public Ray getRay(float2 uv) {
        return Ray(
            position,
            normalize(forward + (2.0 * uv.x - 1.0) * horizontal + (1.0 - 2.0 * uv.y) * vertical),
        );
    }
}

public struct Meshes {
    public StructuredBuffer<float3> positions;
    public StructuredBuffer<float3> normals;
    public StructuredBuffer<uint> indices;
    public StructuredBuffer<MeshDesc> descs;
}

public struct Materials {
    public StructuredBuffer<float> constants;
    public StructuredBuffer<float> variables;
    public RWStructuredBuffer<Atomic<float>> gradient;
    public StructuredBuffer<MaterialDesc> descs;
}

public struct Scene {
    public RaytracingAccelerationStructure tlas;
    public Texture2D<float3> environment_map;
    public SamplerState environment_sampler;

    public PerspectiveCamera camera;
    public Meshes meshes;
    public Materials materials;

    public StructuredBuffer<InstanceDesc> instance_descs;
}

public ParameterBlock<Scene> scene;


