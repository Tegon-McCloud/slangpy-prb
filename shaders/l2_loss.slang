import util;

Texture2D<float4> primal;
Texture2D<float4> reference;
RWStructuredBuffer<Atomic<float>> out;

const float scale;

uint2 dimensions() {
    uint2 dim;
    reference.GetDimensions(dim.x, dim.y);
    return dim;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID) {
    uint2 pixel = tid.xy;

    uint2 dim = dimensions();
    if (pixel.x >= dim.x || pixel.y >= dim.y) {
        return;
    }

    float3 p = primal[pixel].rgb;
    float3 r = reference[pixel].rgb;
    float l = dot(p - r, p - r);

    float l_wave = WaveActiveSum(l);

    if (WaveIsFirstLane()) {
        out[0] += l_wave;
    }
}


