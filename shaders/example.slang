struct Ray {
    float3 origin;
    float3 direction;

    __init(float3 o, float3 d) {
        origin = o;
        direction = d;
    }
}

static const float MAX_DIST = 1e9;
static const float MIN_DIST = 1e-3;

struct Material {
    float roughness;
}

struct SurfacePoint {
    float3 position;
    float3 normal;
} 

struct HitInfo {
    SurfacePoint point;
    Material material;
}

float4 render(int2 id, int2 imageSize) {

    float2 uv = (float2(id.x, imageSize.y - id.y) + 0.5) / float2(imageSize);

    Ray r = generateRay(uv);



    return float4(r.direction, 1.0);
}

Ray generateRay(float2 uv) {
    return Ray(
        float3(0.0, 1.0, 0.0),
        normalize(float3(uv.x - 0.5, -1.0, 0.5 - uv.y))
    );
}

bool traceRay(Ray r, out HitInfo info) {

    float dist = MAX_DIST;
    if (intersectPlane(r, float3(0.0), float3(0.0, 1.0, 0.0), dist, info.point)) {
        info.material = Material(0.5);
    }

    if (dist < MAX_DIST) {
        return true;
    }
}


bool intersectPlane(
    Ray r,
    float3 p,
    float3 n,
    inout float dist,
    out SurfacePoint point
) {
    // (a - p) . n = 0
    // (o - p + td) . n = 0
    // v . n - td . n = 0
    // v . n = td . n
    // t = (v . n) / (d . n)

    float3 v = r.origin - p;
    float t = dot(v, n) / dot(r.direction, n);

    if (t < MIN_DIST || t >= dist) return false;

    dist = t;
    point.position = r.origin + t * r.direction;
    point.normal = n;

    return true;
}
