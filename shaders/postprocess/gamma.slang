implementing postprocess;

public struct Gamma : IPostProcessStage {

    float inv_gamma;

    [Differentiable]
    [NoDiffThis]
    public float3 apply(float3 color) {
        return pow(max(color, 0.0), inv_gamma);
    }
}
