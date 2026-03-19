Texture2D<float4> primal;
Texture2D<float4> reference;
RWTexture2D<float4> adjoint;

const float scale;

float3 l2_adjoint(float3 p, float3 r) {
    return scale * (p - r);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid: SV_DispatchThreadID) {
    uint2 pixel = tid.xy;

    float3 p = primal[pixel].rgb;
    float3 r = reference[pixel].rgb;
    float3 a = l2_adjoint(p, r);

    adjoint[pixel] = float4(a, 1.0);
}


