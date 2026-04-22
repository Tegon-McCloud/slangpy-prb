import util;

public struct MatrixCamera : ICamera {
    public float3 position;
    public float3x3 matrix;

    public Ray getRay(float2 uv) {
        return Ray(
            position,
            mul(matrix, float3(uv, 1.0)),
        );
    }
}

export struct Camera : ICamera = MatrixCamera;