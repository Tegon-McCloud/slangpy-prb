//=================================================================================================
//
//  Baking Lab
//  by MJP and David Neubelt
//  http://mynameismjp.wordpress.com/
//
//  All code licensed under the MIT license
//
//=================================================================================================

struct Tonemapper {
    Texture2D<float4> input;
    RWTexture2D<float4> output;

    void apply_aces_fit(uint2 pixel) {
        float4 color = input[pixel];
        color.rgb = aces_fit(color.rgb);
        output[pixel] = color;
    }

}

// sRGB => XYZ => D65_2_D60 => AP1 => RRT_SAT
static const float3x3 input_transform =
{
    {0.59719, 0.35458, 0.04823},
    {0.07600, 0.90834, 0.01566},
    {0.02840, 0.13383, 0.83777}
};

// ODT_SAT => XYZ => D60_2_D65 => sRGB
static const float3x3 output_transform =
{
    { 1.60475, -0.53108, -0.07367},
    {-0.10208,  1.10813, -0.00605},
    {-0.00327, -0.07276,  1.07602}
};

float3 rrt_odt_fit(float3 v)
{
    float3 a = v * (v + 0.0245786) - 0.000090537;
    float3 b = v * (0.983729 * v + 0.4329510f) + 0.23808;
    return a / b;
}

float3 aces_fit(float3 color)
{
    color = mul(input_transform, color);
    color = rrt_odt_fit(color);
    color = mul(output_transform, color);
    color = saturate(color);

    return color;
}

ParameterBlock<Tonemapper> tonemapper;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid: SV_DispatchThreadID) {
    tonemapper.apply_aces_fit(tid.xy);
}

