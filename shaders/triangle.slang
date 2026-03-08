import util;
import scene;

[shader("closesthit")]
void closest_hit_triangle_primary(inout RayTraceResponse response, BuiltInTriangleIntersectionAttributes attribs)
{
    HitDesc hit;

    uint instance_index = InstanceIndex();
    InstanceDesc instance_desc = scene.instance_descs[instance_index];
    MeshDesc mesh_desc = scene.mesh_descs[instance_desc.mesh_index];

    uint primitive_index = PrimitiveIndex();

    uint i0 = scene.indices[mesh_desc.index_offset + primitive_index * 3 + 0];
    uint i1 = scene.indices[mesh_desc.index_offset + primitive_index * 3 + 1];
    uint i2 = scene.indices[mesh_desc.index_offset + primitive_index * 3 + 2];

    float3 n0 = scene.normals[mesh_desc.vertex_offset + i0];
    float3 n1 = scene.normals[mesh_desc.vertex_offset + i1];
    float3 n2 = scene.normals[mesh_desc.vertex_offset + i2];

    float3 barycentrics = float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y, attribs.barycentrics.x, attribs.barycentrics.y);

    float3 n_object = normalize(barycentrics.x * n0 + barycentrics.y * n1 + barycentrics.z * n2);
    float3 n = mul(ObjectToWorld(), float4(n_object, 0.0));

    hit.distance = RayTCurrent();
    hit.tangent_to_world = make_tbn_matrix(n);
    hit.material_index = instance_desc.material_index;

    response.isHit = true;
    response.hit = hit;
}

[shader("anyhit")]
void any_hit_triangle_occlusion(inout bool occluded) {
    occluded = true;
}
