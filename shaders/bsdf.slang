module bsdf;
__include bsdf.lambertian;
__include bsdf.microfacet;

import util;
import random;

public struct BsdfEvaluation {
    /*
        `parameters` pointer is reinterpret cast to a `uint64_t`, since
        pipeline creation fails when a pointer is passed to a callable
        shader.
    */ 
    public uint64_t parameters; 
    public uint rand_state;
    public float3 wo;
    public float3 wi;
    public float3 value;

    [mutating]
    public void set_parameter_ptr(Ptr<void, Access.Read, AddressSpace.Device> parameters) {
        this.parameters = reinterpret<uint64_t>(parameters);
    }

    public T get_parameters<T>() {
        return *reinterpret<T *>(this.parameters);
    }

}

public struct BsdfSample {
    /*
        `parameters` pointer is reinterpret cast to a `uint64_t`, since
        pipeline creation fails when a pointer is passed to a callable
        shader.
    */
    public uint64_t parameters;
    public uint rand_state;
    public float3 wo;
    public float3 wi;
    public float3 value;
    public float pdf;

    [mutating]
    public void set_parameter_ptr(Ptr<void, Access.Read, AddressSpace.Device> parameters) {
        this.parameters = reinterpret<uint64_t>(parameters);
    }

    public T get_parameters<T>() {
        return *reinterpret<T *>(this.parameters);
    }
}
