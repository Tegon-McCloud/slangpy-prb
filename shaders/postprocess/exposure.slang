
implementing postprocess;

public struct Exposure : IPostProcessStage {

    float scale;

    [Differentiable]
    [NoDiffThis]
    public float3 apply(float3 color) {
        return scale * color;
    }
}
