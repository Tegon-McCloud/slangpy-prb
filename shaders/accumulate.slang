
struct Accumulator {
    Texture2D sample;
    RWTexture2D accumulation;

    void execute(uint2 pixel) {

        float4 s = sample[pixel];
        float4 a = accumulation[pixel];

        a += float4(s.xyz, 1.0);

        accumulation[pixel] = a;
    }

}

ParameterBlock<Accumulator> accumulator;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid: SV_DispatchThreadID) {
    accumulator.execute(tid.xy);
}
