struct GradientDescent {
    const float learning_rate;

    void step<let n : int>(
        inout vector<float, n> v,
        float g,
    ) {
        v -= learning_rate * g;
    }
}

ParameterBlock<GradientDescent> gradient_descent;

RWStructuredBuffer<float> variables;
StructuredBuffer<float> gradient;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid: SV_DispatchThreadID) {
    uint index = tid.x;

    float1 v = variables[index];
    float1 g = gradient[index];

    gradient_descent.step(v, g,);

    variables[index] = v;
}
