import util;

struct Adam {
    const float learning_rate;
    const float beta1;
    const float scale1; // = 1 / (1 - beta1^step)
    const float beta2;
    const float scale2; // = 1 / (1 - beta2^step)
    const float epsilon;

    void step<let n : int>(
        inout vector<float, n> v,
        float g,
        inout vector<float, n> m1,
        inout vector<float, n> m2,
    ) {
        m1 = lerp(g, m1, beta1);
        m2 = lerp(sq(g), m2, beta2);

        vector<float, n> m1_scaled = scale1 * m1;
        vector<float, n> m2_scaled = scale2 * m2;

        v -= learning_rate * m1_scaled / (sqrt(m2_scaled) + epsilon);
    }
}

ParameterBlock<Adam> adam;

RWStructuredBuffer<float> variables;
StructuredBuffer<float> gradient;
RWStructuredBuffer<float> moment1;
RWStructuredBuffer<float> moment2;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid: SV_DispatchThreadID) {
    uint index = tid.x;

    float1 v = variables[index];
    float1 g = gradient[index];
    float1 m1 = moment1[index];
    float1 m2 = moment2[index];

    adam.step(v, g, m1, m2);

    variables[index] = v;
    moment1[index] = m1;
    moment2[index] = m2;
}
