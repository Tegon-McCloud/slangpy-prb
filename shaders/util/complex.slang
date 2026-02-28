implementing util;

public struct Complex {
    public float real;
    public float imag;
}

public Complex operator +(Complex a, Complex b) {
    return Complex(a.real + b.real, a.imag + b.imag);
}

public Complex operator -(Complex a, Complex b) {
    return Complex(a.real - b.real, a.imag - b.imag);
}

public Complex operator *(Complex a, Complex b) {
    return Complex(a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real);
}

public Complex operator /(Complex a, Complex b) {
    float s = 1.0 / (b.real * b.real + b.imag * b.imag);
    return Complex(s * (a.real * b.real + a.imag * b.imag), s * (a.imag * b.real - a.real * b.imag));
}

public Complex operator +(Complex a, float b) {
    return Complex(a.real + b, a.imag);
}

public Complex operator +(float a, Complex b) {
    return Complex(a + b.real, b.imag);
}

public Complex operator -(Complex a, float b) {
    return Complex(a.real - b, a.imag);
}

public Complex operator -(float a, Complex b) {
    return Complex(a - b.real, -b.imag);
}

public Complex operator *(Complex a, float b) {
    return Complex(a.real * b, a.imag * b);
}

public Complex operator *(float a, Complex b) {
    return Complex(a * b.real, a * b.imag);
}

public Complex operator /(Complex a, float b) {
    return Complex(a.real / b, a.imag / b);
}

public Complex operator /(float a, Complex b) {
    float s = 1.0 / (b.real * b.real + b.imag * b.imag);
    return Complex(s * (a * b.real), s * (-a * b.imag));
}

public Complex conj(Complex z) {
    return Complex(z.real, -z.imag);
}

public float abs_sq(Complex z) {
    return z.real * z.real + z.imag * z.imag;
}

public float abs(Complex z) {
    return sqrt(z.real * z.real + z.imag * z.imag);
}

public Complex sqrt(Complex z) {
    float n = abs(z);

    if (n == 0.0) {
        return Complex(0.0, 0.0);
    }

    float t1 = sqrt(0.5 * (n + abs(z.real)));
    float t2 = 0.5 * z.imag / t1;

    if (z.real >= 0.0) {
        return Complex(t1, t2);
    } else {
        return Complex(abs(t2), copysign(t1, z.imag));
    }
}
