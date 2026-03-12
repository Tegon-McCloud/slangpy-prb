module bsdf;
__include bsdf.lambertian;
// __include bsdf.specular_conductor;
// __include bsdf.specular_dielectric;
// __include bsdf.microfracet_conductor;
__include bsdf.microfacet_dielectric;

import util;

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
    public uint error_state;
    public float3 wo;
    public float ior_outside;
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

public struct BsdfBackpropagation {
    /*
        `parameters` and `gradients` pointers is reinterpret cast to a `uint64_t`, since
        pipeline creation fails when a pointer is passed to a callable
        shader.
    */
    public uint64_t parameters;
    public uint64_t gradients;
    public float3 wo;
    public float3 wi;
    public float3 weight;
    
    [mutating]
    public void set_parameter_ptr(Ptr<void, Access.Read, AddressSpace.Device> parameters) {
        this.parameters = reinterpret<uint64_t>(parameters);
    }

    public T get_parameters<T>() {
        return *reinterpret<T *>(this.parameters);
    }

    [mutating]
    public void set_gradients_ptr(Ptr<void, Access.ReadWrite, AddressSpace.Device> gradients) {
        this.gradients = reinterpret<uint64_t>(gradients);
    }

    public Ptr<T, Access.ReadWrite, AddressSpace.Device> get_gradients<T>() {
        return reinterpret<Ptr<T, Access.ReadWrite, AddressSpace.Device>>(this.gradients);
    }
}
