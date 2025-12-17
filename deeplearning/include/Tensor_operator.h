#ifndef TENSOR_OPERATOR_H
#define TENSOR_OPERATOR_H

#include "Tensor.h"
#include <cmath>

/*
Ini biar bisa support operator aritmatika dengan Tensor!
Jadi ini gak akan ada error dan membantu dalam perhitungan,
neural network si Tensor nya!
*/

// TENSOR + Operasi Tensor //
// Tensor + Tensor //
inline Tensor operator+(Tensor lhs, const Tensor& rhs) {
    return lhs += rhs;
}

// Tensor - Tensor //
inline Tensor operator-(Tensor lhs, const Tensor& rhs) {
    return lhs -= rhs;
}

// Tensor * Tensor (element-wise) //
inline Tensor operator*(Tensor lhs, const Tensor& rhs) {
    return lhs *= rhs;
}

// Tensor / Tensor //
inline Tensor operator/(Tensor lhs, const Tensor& rhs) {
    return lhs /= rhs;
}

// SCALAR - Operasi Tensor //
// scalar + Tensor //
inline Tensor operator+(double s, const Tensor& t) {
    Tensor out(t.get_shape());
    for (int i = 0; i < t.numel(); ++i)
        out[i] = s + t[i];
    return out;
}

// Tensor + scalar //
inline Tensor operator+(Tensor t, double s) {
    return t += s;
}

// scalar - Tensor //
inline Tensor operator-(double s, const Tensor& t) {
    Tensor out(t.get_shape());
    for (int i = 0; i < t.numel(); ++i)
        out[i] = s - t[i];
    return out;
}

// Tensor - scalar //
inline Tensor operator-(Tensor t, double s) {
    return t -= s;
}

// scalar * Tensor //
inline Tensor operator*(double s, const Tensor& t) {
    Tensor out(t.get_shape());
    for (int i = 0; i < t.numel(); ++i)
        out[i] = s * t[i];
    return out;
}

// Tensor * scalar //
inline Tensor operator*(Tensor t, double s) {
    return t *= s;
}

// scalar / Tensor //
inline Tensor operator/(double s, const Tensor& t) {
    Tensor out(t.get_shape());
    for (int i = 0; i < t.numel(); ++i)
        out[i] = s / t[i];
    return out;
}

// Tensor / scalar //
inline Tensor operator/(Tensor t, double s) {
    return t /= s;
}

// Fungsi matematika //
// exp function untuk Tensor //
inline Tensor exp(const Tensor& t) {
    Tensor out(t.get_shape());
    for (int i = 0; i < t.numel(); ++i)
        out[i] = std::exp(t[i]);
    return out;
}

// sqrt function untuk Tensor (element-wise) //
inline Tensor sqrt(const Tensor& t) {
    Tensor out(t.get_shape());
    for (int i = 0; i < t.numel(); ++i)
        out[i] = std::sqrt(t[i]);
    return out;
}

// log function untuk Tensor (element-wise) //
inline Tensor log(const Tensor& t) {
    Tensor out(t.get_shape());
    for (int i = 0; i < t.numel(); ++i)
        out[i] = std::log(t[i]);
    return out;
}

#endif