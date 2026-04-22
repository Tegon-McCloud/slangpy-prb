
implementing postprocess;

public struct SrgbEncoder : IPostProcessStage {

    [Differentiable]
    public float3 apply(float3 color) {
        return select(
            color < 0.0031308,
            12.92 * color,
            1.055 * pow(color, 1.0 / 2.4) - 0.055,
        );
    }
}
