import util;
import scene;

[shader("miss")]
void miss_primary(inout RayTraceResponse response)
{
    float3 w = WorldRayDirection();

    float phi = atan2(w.z, w.x) + pi;
    float theta = acos(w.y);

    float2 uv = float2(phi / tau, theta / pi);

    response.isHit = false;
    response.miss.radiance = scene.environment_map.Sample(scene.environment_sampler, uv).rgb;
}

[shader("miss")]
void miss_occlusion()
{}
