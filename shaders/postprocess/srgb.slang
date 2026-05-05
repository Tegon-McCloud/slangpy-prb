
implementing postprocess;

public struct SrgbEncoder : IPostProcessStage {

    float inv_gamma;

    [Differentiable]
    [NoDiffThis]
    public float3 apply(float3 color) {
        return select(
            color < 0.0031308,
            12.92 * color,
            1.055 * pow(color, inv_gamma) - 0.055,
        );
    }
}
