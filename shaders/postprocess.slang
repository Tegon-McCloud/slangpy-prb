module postprocess;

__include postprocess.exposure;
__include postprocess.tonemap;
__include postprocess.srgb;

public interface IPostProcessStage {
    [Differentiable]
    [NoDiffThis]
    public float3 apply(float3 color);
}
