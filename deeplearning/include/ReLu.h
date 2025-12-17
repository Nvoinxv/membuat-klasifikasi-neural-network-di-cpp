#ifndef RELU_H
#define RELU_H

#include "Tensor.h"
#include "Tensor_operator.h"
#include <algorithm>

/*
Kegunaan relu ini adalah untuk membuat perhitungan aktivasi
menjadi lebih efektif. ReLu ini jika ada bilangan negatif maka akan di ubah jadi 0,
sementara bilangan positif akan di pertahankan. Maka nya ini cocok untuk non linear
biar gak kaku output si AI atau pelatihan nya.
*/

/*
Rumus matematika ReLu untuk forward nya seperti ini:
F (x) = max (0, x)
Lalu untuk backward nya seperti ini:
F' (x) = 1 jika x > 0, 0 jika x <= 0
*/

class ReLu {
    public:
    // Forward pass //
    static Tensor forward(const Tensor& x) {
        Tensor out(x.get_shape());
        for (int i = 0; i < x.numel(); i++) {
            out[i] = std::max(0.0, x[i]);  
        }
        return out;
    }

    // Backward pass //
    static Tensor backward(const Tensor& x) {
        Tensor out(x.get_shape());
        for (int i = 0; i < x.numel(); i++) {
            out[i] = (x[i] > 0) ? 1.0 : 0.0;
        }
        return out;
    }
};

#endif