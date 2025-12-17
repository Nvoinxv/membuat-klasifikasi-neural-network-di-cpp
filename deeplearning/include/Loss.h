#ifndef LOSS_H
#define LOSS_H

#include "Tensor.h"
#include "Tensor_operator.h"
#include "Tensor_factory.h"
#include <algorithm>

/*
Apa sih itu binarycross entropy?
Ini adalah metode mengukur loss atau kerugian selama pelatihan AI,
Biar bisa di pakai untuk gradient dan update bobot.
Soal nya dalam backward, si loss ini yang di gunakan di awal awal.
*/

/*
Rumus binary cros entropy seperti ini:
BCE = - (y * log(y_pred) + (1 - y) * log(1 - y_pred))
Yang dimana y_pred adalah prediksi model dan y adalah label aktual.
Ini konteks nya harus 0 - 1 karna dari nama rumus nya saja binary cross entropy.

Lalu untuk backward nya seperti ini perhitungan nya:
Backward BCE = (y_pred - y_test) / (y_pred *(1-y_pred))
*/

class BinaryCrossEnrtopy {
public:
    // Forward pass dengan numerical stability //
    // Kita clamp y_pred supaya tidak ada log(0) = -inf //
    static Tensor forward(const Tensor& y_pred, const Tensor& y_test) {
        Tensor out(y_pred.get_shape());
        const double eps = 1e-7;  // Small epsilon untuk numerical stability //
        
        for (int i = 0; i < y_pred.numel(); ++i) {
            // Clamp y_pred antara eps dan 1-eps untuk menghindari log(0) //
            double p = std::max(eps, std::min(1.0 - eps, y_pred[i]));
            double y = y_test[i];
            out[i] = -(y * std::log(p) + (1.0 - y) * std::log(1.0 - p));
        }
        return out;
    }

    // Backward pass dengan numerical stability //
    static Tensor backward(const Tensor& y_pred, const Tensor& y_test) {
        Tensor out(y_pred.get_shape());
        const double eps = 1e-7;  // Small epsilon untuk numerical stability //
        
        for (int i = 0; i < y_pred.numel(); ++i) {
            // Clamp y_pred untuk menghindari division by zero //
            double p = std::max(eps, std::min(1.0 - eps, y_pred[i]));
            double y = y_test[i];
            // Gradient: (p - y) / (p * (1 - p)) //
            out[i] = (p - y) / (p * (1.0 - p));
        }
        return out;
    }
};

#endif