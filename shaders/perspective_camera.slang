import util;

public struct PerspectiveCamera : ICamera {
    public float3 forward;
    public float3 horizontal;
    public float3 vertical;
    public float3 position;

    public Ray getRay(float2 uv) {
        return Ray(
            position,
            normalize(forward + (2.0 * uv.x - 1.0) * horizontal + (1.0 - 2.0 * uv.y) * vertical),
        );
    }
}

export struct Camera : ICamera = PerspectiveCamera;
