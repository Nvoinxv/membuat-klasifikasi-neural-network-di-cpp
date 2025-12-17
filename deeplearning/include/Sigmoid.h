#ifndef SIGMOID_H
#define SIGMOID_H

/*
Kita membuat fungsi sigmoid.
Dimana fungsi ini sangat berguna untuk binary classification.
Karna disini kita fokus klasifikasi klasik, jadi kita pakai fungsi ini saja.
*/

#include "Tensor.h"
#include "Tensor_operator.h"
#include <cmath>

class Sigmoid {
    public:
    static Tensor forward(const Tensor& x) {
        // Inti perhitungan sigmoid forward nya //
        return 1.0 / (1.0 + exp(-x));
    };

    static Tensor backward(const Tensor& x) {
        // Ini perhitungan backward nya //
        return x * (1.0 - x);
    };
};

#endif 

