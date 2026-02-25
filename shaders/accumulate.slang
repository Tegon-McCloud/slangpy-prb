
struct Accumulator {
    Texture2D sample;
    RWTexture2D accumulation;

    void accumulate(uint2 pixel) {

        float4 s = sample[pixel];
        float4 a = accumulation[pixel];

        if (!all(isfinite(s))) {
            printf("%i, %i, %i", pixel.x, pixel.y, (int)a.w);
            return;
        }

        a += float4(s.xyz, 1.0);

        accumulation[pixel] = a;
    }
}

ParameterBlock<Accumulator> accumulator;

[shader("compute")]
[numthreads(8, 8, 1)]
void accumulate(uint3 tid: SV_DispatchThreadID) {
    accumulator.accumulate(tid.xy);
}
