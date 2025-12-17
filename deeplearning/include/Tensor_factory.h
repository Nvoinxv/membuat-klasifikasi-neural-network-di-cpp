#ifndef TENSOR_FACTORY_H
#define TENSOR_FACTORY_H

#include "Tensor.h"
#include "Tensor_operator.h"
#include <random>
#include <chrono>
#include <initializer_list>
#include <sstream>
#include <iomanip>

/*
Ini adalah bagian tensor factory, biar siap di pakai deep learning.
Ini gw ambil inspirasi dari Pytorch, cuma gw buat versi C++.
*/

// namespace dl ini adalah pemanggilan biar lebih mudah saja //
namespace dl {

// RANDOM NUMBER GENERATOR //
// Global random engine //
inline std::mt19937& get_random_engine() {
    static std::mt19937 gen(
        static_cast<unsigned int>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count()
        )
    );
    return gen;
}

// Set random seed (untuk reproducibility)
inline void manual_seed(unsigned int seed) {
    get_random_engine().seed(seed);
}

// BASIC FACTORY FUNCTIONS //
// Buat tensor dari shape dengan nilai 0
inline Tensor zeros(const std::vector<int>& shape) {
    Tensor t(shape);
    for (int i = 0; i < t.numel(); ++i) {
        t[i] = 0.0;
    }
    return t;
}

// Buat tensor dari shape dengan nilai 1
inline Tensor ones(const std::vector<int>& shape) {
    Tensor t(shape);
    for (int i = 0; i < t.numel(); ++i) {
        t[i] = 1.0;
    }
    return t;
}

// Buat tensor dengan nilai tertentu
inline Tensor full(const std::vector<int>& shape, double value) {
    Tensor t(shape);
    for (int i = 0; i < t.numel(); ++i) {
        t[i] = value;
    }
    return t;
}

// RANDOM TENSOR FUNCTIONS //
// Tensor dengan random uniform distribution [0, 1] //
inline Tensor rand(const std::vector<int>& shape) {
    Tensor t(shape);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < t.numel(); ++i) {
        t[i] = dist(get_random_engine());
    }
    return t;
}

// Tensor dengan random normal distribution (mean=0, std=1) //
inline Tensor randn(const std::vector<int>& shape) {
    Tensor t(shape);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < t.numel(); ++i) {
        t[i] = dist(get_random_engine());
    }
    return t;
}

// Tensor dengan random uniform dalam range [low, high] //
inline Tensor uniform(const std::vector<int>& shape, double low, double high) {
    Tensor t(shape);
    std::uniform_real_distribution<double> dist(low, high);
    for (int i = 0; i < t.numel(); ++i) {
        t[i] = dist(get_random_engine());
    }
    return t;
}

// Tensor dengan random normal dengan mean dan std custom //
inline Tensor normal(const std::vector<int>& shape, double mean, double std) {
    Tensor t(shape);
    std::normal_distribution<double> dist(mean, std);
    for (int i = 0; i < t.numel(); ++i) {
        t[i] = dist(get_random_engine());
    }
    return t;
}

// SEQUENCE TENSOR FUNCTIONS //
// Buat tensor berurutan [start, end] dengan step //
inline Tensor arange(double start, double end, double step = 1.0) {
    std::vector<double> data;
    for (double val = start; val < end; val += step) {
        data.push_back(val);
    }
    return Tensor({static_cast<int>(data.size())}, data);
}

// Buat tensor dengan n nilai linear dari start sampai end //
inline Tensor linspace(double start, double end, int steps) {
    std::vector<double> data(steps);
    if (steps == 1) {
        data[0] = start;
    } else {
        double step = (end - start) / (steps - 1);
        for (int i = 0; i < steps; ++i) {
            data[i] = start + i * step;
        }
    }
    return Tensor({steps}, data);
}

// IDENTITY & SPECIAL TENSORS //
// Buat identity matrix (untuk linear algebra) //
inline Tensor eye(int n) {
    Tensor t({n, n});
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            t.at({i, j}) = (i == j) ? 1.0 : 0.0;
        }
    }
    return t;
}

// Buat diagonal matrix dari vector //
inline Tensor diag(const std::vector<double>& values) {
    int n = static_cast<int>(values.size());
    Tensor t({n, n});
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            t.at({i, j}) = (i == j) ? values[i] : 0.0;
        }
    }
    return t;
}

// inisailisasi list factory //

/*
Ini gw buat biar lebih mudah di panggil dan gampang di pakai.
Tanpa lu harus manual manual, cukup panggil Tensor::tensor().
Maka nya gw terinspirasi dari pytorch, tensorflow, atau gak numpy!
*/

// 1D Tensor dari list inisialisasi //
inline Tensor tensor(std::initializer_list<double> data) {
    std::vector<double> vec(data);
    return Tensor({static_cast<int>(vec.size())}, vec);
}

// 2D Tensor dari list inisialisasi  //
inline Tensor tensor(std::initializer_list<std::initializer_list<double>> data) {
    int rows = static_cast<int>(data.size());
    int cols = static_cast<int>(data.begin()->size());
    
    std::vector<double> flat_data;
    flat_data.reserve(rows * cols);
    
    for (const auto& row : data) {
        for (double val : row) {
            flat_data.push_back(val);
        }
    }
    
    return Tensor({rows, cols}, flat_data);
}

// 3D Tensor dari ketiga nested list inisialisasi  //
inline Tensor tensor(std::initializer_list<std::initializer_list<std::initializer_list<double>>> data) {
    int dim0 = static_cast<int>(data.size());
    int dim1 = static_cast<int>(data.begin()->size());
    int dim2 = static_cast<int>(data.begin()->begin()->size());
    
    std::vector<double> flat_data;
    flat_data.reserve(dim0 * dim1 * dim2);
    
    for (const auto& matrix : data) {
        for (const auto& row : matrix) {
            for (double val : row) {
                flat_data.push_back(val);
            }
        }
    }
    
    return Tensor({dim0, dim1, dim2}, flat_data);
}

// Tensor dari vector (1D) //
inline Tensor tensor(const std::vector<double>& data) {
    return Tensor({static_cast<int>(data.size())}, data);
}

// Helper untuk column vector (2D tensor dengan 1 kolom) //
inline Tensor column_vector(std::initializer_list<double> data) {
    std::vector<double> vec(data);
    return Tensor({static_cast<int>(vec.size()), 1}, vec);
}

// Atau kalo mau lebih explicit //
inline Tensor tensor_2d_col(std::initializer_list<double> data) {
    std::vector<double> vec(data);
    return Tensor({static_cast<int>(vec.size()), 1}, vec);
}

// Tensor dari shape + data //
/*
Tensor ini akan di simpan sebagai vektor data.
Yang diantara nya ada shape dan data.
contoh disini input nya kek gini: 
{{1,2},{3,4}}

maka output nya disimpan jadi seperti ini:
shape = {2, 2}
data  = {1, 2, 3, 4}
*/
inline Tensor tensor(const std::vector<int>& shape, const std::vector<double>& data) {
    return Tensor(shape, data);
}

// Inisalisasi Bobot //
/*
Apa kegunaan xavier tensor?
Kegunaan ini untuk mengisi nilai awal bobot tensor dengan nilai random.
Jadi dengan ada nya xavier tensor, membuat aliran sinyal atau neural network jadi stabil.
Semisal bobot nya kecil tanpa xavier maka bobot akan menjadi menghilang atau malah nol,
sedangkan jika bobot nya besar maka makin lama makin meledak.
Akibat nya model tidak dapat belajar secara efektif dan akurasi yang bagus.
*/

// Xavier/Glorot Initialization //
inline Tensor xavier_uniform(const std::vector<int>& shape) {
    // Asumsi shape = {fan_out, fan_in} untuk fully connected //
    int fan_in = (shape.size() >= 2) ? shape[1] : shape[0];
    int fan_out = shape[0];
    
    // Ini adalah rumus perhitungan xavier uniform //
    // arti nya makin besar, rentang bobot semakin kecil //
    // kalau kecil maka rentang bobot semakin besar //
    double limit = std::sqrt(6.0 / (fan_in + fan_out));
    return uniform(shape, -limit, limit);
}

inline Tensor xavier_normal(const std::vector<int>& shape) {
    int fan_in = (shape.size() >= 2) ? shape[1] : shape[0];
    int fan_out = shape[0];
    
    double std = std::sqrt(2.0 / (fan_in + fan_out));
    return normal(shape, 0.0, std);
}

// Kaiming/He Initialization //
inline Tensor kaiming_uniform(const std::vector<int>& shape) {
    int fan_in = (shape.size() >= 2) ? shape[1] : shape[0];
    
    double limit = std::sqrt(6.0 / fan_in);
    return uniform(shape, -limit, limit);
}

inline Tensor kaiming_normal(const std::vector<int>& shape) {
    int fan_in = (shape.size() >= 2) ? shape[1] : shape[0];
    
    double std = std::sqrt(2.0 / fan_in);
    return normal(shape, 0.0, std);
}

// Operasi Tensor  //

// Clone tensor (deep copy) //
inline Tensor clone(const Tensor& t) {
    return Tensor(t.get_shape(), t.get_data());
}

// Zeros like (sama shape, isi 0) //
inline Tensor zeros_like(const Tensor& t) {
    return zeros(t.get_shape());
}

// Ones like (sama shape, isi 1) //
inline Tensor ones_like(const Tensor& t) {
    return ones(t.get_shape());
}

// Rand like (sama shape, random uniform) //
inline Tensor rand_like(const Tensor& t) {
    return rand(t.get_shape());
}

// Randn like (sama shape, random normal) //
inline Tensor randn_like(const Tensor& t) {
    return randn(t.get_shape());
}

} // namespace dl //

// PRINT TENSOR //
// Overload operator<< untuk print tensor ke cout //
inline std::ostream& operator<<(std::ostream& os, const Tensor& t) {
    const auto& shape = t.get_shape();
    const auto& data = t.get_data();
    
    os << "Tensor(shape=[";
    for (size_t i = 0; i < shape.size(); ++i) {
        os << shape[i];
        if (i < shape.size() - 1) os << ", ";
    }
    os << "], data=";
    
    if (shape.size() == 1) {
        // 1D Tensor //
        os << "[";
        for (int i = 0; i < t.numel(); ++i) {
            os << std::fixed << std::setprecision(4) << data[i];
            if (i < t.numel() - 1) os << ", ";
        }
        os << "]";
    } 
    else if (shape.size() == 2) {
        // 2D Tensor (Matrix) //
        os << "\n[";
        for (int i = 0; i < shape[0]; ++i) {
            os << "[";
            for (int j = 0; j < shape[1]; ++j) {
                os << std::fixed << std::setprecision(4) << t.at({i, j});
                if (j < shape[1] - 1) os << ", ";
            }
            os << "]";
            if (i < shape[0] - 1) os << ",\n ";
        }
        os << "]";
    }
    else {
        // dimensi tertinggi, hanya menampilkan data sebanyak 10 elemen//
        os << "[";
        int show_count = std::min(10, t.numel());
        for (int i = 0; i < show_count; ++i) {
            os << std::fixed << std::setprecision(4) << data[i];
            if (i < show_count - 1) os << ", ";
        }
        if (t.numel() > 10) os << ", ...";
        os << "]";
    }
    
    os << ")";
    return os;
}

#endif
