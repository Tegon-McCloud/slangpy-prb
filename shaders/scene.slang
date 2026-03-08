import util;

public struct MeshDesc {
    public uint vertex_offset;
    public uint index_offset;
}

public struct MaterialDesc {
    public uint evaluate_call_index;
    public uint sample_call_index;
    public uint backpropagate_call_index;
    public uint requires_grad;
    public Ptr<void, Access.Read, AddressSpace.Device> parameters;
    public Ptr<void, Access.ReadWrite, AddressSpace.Device> gradients;
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

public struct Scene {
    public RaytracingAccelerationStructure tlas;
    public Texture2D<float3> environment_map;
    public SamplerState environment_sampler;

    public PerspectiveCamera camera;

    public StructuredBuffer<float3> positions;
    public StructuredBuffer<float3> normals;
    public StructuredBuffer<uint> indices;
    public StructuredBuffer<MeshDesc> mesh_descs;

    public StructuredBuffer<MaterialDesc> material_descs;

    public StructuredBuffer<InstanceDesc> instance_descs;

    private StructuredBuffer<uint8_t> material_parameters;
    private RWStructuredBuffer<uint8_t> material_gradients; 
}

public ParameterBlock<Scene> scene;
