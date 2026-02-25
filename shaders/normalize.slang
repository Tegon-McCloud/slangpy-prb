struct Accumulator {
    RWTexture2D accumulation;

    void normalize(uint2 pixel) {
        float4 a = accumulation[pixel];
        a /= a.w;
        accumulation[pixel] = a;
    }
}

ParameterBlock<Accumulator> accumulator;

[shader("compute")]
[numthreads(8, 8, 1)]
void normalize(uint3 tid: SV_DispatchThreadID) {
    accumulator.normalize(tid.xy);
}
