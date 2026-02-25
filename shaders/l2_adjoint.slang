
Texture2D<float4> primal;
Texture2D<float4> reference;
RWTexture2D<float4> adjoint;

float4 l2_adjoint(float4 p, float4 r) {
    return 2.0 * (p - r);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid: SV_DispatchThreadID) {
    uint2 pixel = tid.xy;

    float4 p = primal[pixel];
    float4 r = reference[pixel];
    float4 a = l2_adjoint(p, r);

    adjoint[pixel] = float4(a.xyz, 1.0);
}


