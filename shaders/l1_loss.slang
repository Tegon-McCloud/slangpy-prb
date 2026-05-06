import util;

Texture2D<float4> reference;

float sum_channels(float3 v) {
    return v.x + v.y + v.z;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void loss(
    uint3 tid: SV_DispatchThreadID,
    uniform Texture2D<float4> primal,
    uniform RWStructuredBuffer<Atomic<float>> output,
) {
    uint2 pixel = tid.xy;

    uint2 dim = dimensions(reference);
    if (pixel.x >= dim.x || pixel.y >= dim.y) {
        return;
    }

    float3 p = primal[pixel].rgb;
    float3 r = reference[pixel].rgb;
    float l = sum_channels(abs(p - r));

    float l_wave = WaveActiveSum(l);

    if (WaveIsFirstLane()) {
        output[0] += l_wave;
    }
}

[shader("compute")]
[numthreads(16, 16, 1)]
void backwards(
    uint3 tid: SV_DispatchThreadID,
    uniform Texture2D<float4> primal,
    uniform RWTexture2D<float4> gradient,
    uniform float scale,
) {
    uint2 pixel = tid.xy;

    uint2 dim = dimensions(primal);
    if (pixel.x >= dim.x || pixel.y >= dim.y) {
        return;
    }

    float3 p = primal[pixel].rgb;
    float3 r = reference[pixel].rgb;
    float3 g = scale * sign(p - r);

    gradient[pixel] = float4(g, 1.0);
}

