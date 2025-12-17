#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <iostream>
#include <numeric>
#include <cassert>
#include <cmath>

/*
Apa sih itu Tensor?
Jadi Tensor adalah suatu array yang dapat berdimensi lebih tinggi.
Ini di temukan di pelajaran Aljabar Linear.
Konsep nya sama kek Tensor 0 itu skalar, Tensor 1 itu vektor,
Tensor 2 itu matriks, dan Terakhir Tensor 3 itu Tensor.
Jadi dengan ada nya Tensor, membuat perhitungan deeplearning jadi fleksibel,
dengan banyak nya dimensi. I mean kalau dimensi A kecil bertemu dimensi B yg besar,
maka tetap bisa di hitung dengan ada bantuan Tensor ini.
Teknik Tensor yang gw pakai itu broadcasting, jadi broadcasting ini kek meregangkan array.
Jadi kalau dimensi A kecil bertemu dimensi B yang besar, maka dimensi A akan di meregangkan
sesuai dengan dimensi B yang besar. Contoh misal nya A = (2,3) dan B itu (4,2,3), maka A akan di meregangkan menjadi (4,2,3)
untuk bisa di hitung dengan B. */

class Tensor {
    private:
    // Ini variabel yang wajib di pakai dalam pembuatan Tensor //

    /* Ingat mindset dalam Tensor.
        Tensor = 1D array + metadata
    */ 

    std::vector<double> data;
    std::vector<int> bentuk;
    std::vector<int> strides;
    bool requires_grad;

    public:

    // Kita membuat fungsi hitung jumlah elemen atau size elemen dalam Tensor //
    int numel(const std::vector<int>& bentuk) const {
        return std::accumulate(bentuk.begin(), bentuk.end(), 1, std::multiplies<int>());
    };

    // Versi numel tanpa argumen - menggunakan bentuk internal //
    int numel() const {
        return std::accumulate(bentuk.begin(), bentuk.end(), 1, std::multiplies<int>());
    };

    // Getter untuk bentuk/shape //
    const std::vector<int>& get_shape() const {
        return bentuk;
    };

    // Getter untuk data (read-only) //
    const std::vector<double>& get_data() const {
        return data;
    };

    // Ini bagian Inti Tensor //
    // Lu perlu buat perhitungan strides //
    // Jadi gampang nya itu strides kek "Kalau indeks di dimensi ini naik 1, lompat berapa di memori?" //

    std::vector<int> perhitungan_strides(const std::vector<int>& bentuk) const {
        // Cek semua ukuran elemen //
        int n = bentuk.size();
        // Inisialisasi strides //
        std::vector<int> strides(n);
        if (n > 0) {
            // Lalu set elemen terakhir menjadi 1 //
            strides[n-1] = 1;
            
            // Melakukan pengulangan atau while loop //
            for (int i = n - 2; i >= 0; --i) {
                strides[i] = strides[i+1] * bentuk[i+1];
            };
        }

        return strides;
    };

    // Membuat konstruktor Tensor //

    /* Apa sih itu konstruktor Tensor? 
    Konstruktor Tensor itu kek "Kalau ada Tensor baru, lakukan apa dulu?"
    Ini yang membuat Tensor bisa multi dimensi.
    */ 

    // Default constructor
    Tensor() : requires_grad(false) {};

    // Ini bagian paling penting dalam multi dimensi setelah perhitungan strides //
    Tensor(const std::vector<int>& bentuk_) {
        // inisialisasi bentuk dan strides //
        bentuk = bentuk_;
        strides = perhitungan_strides(bentuk);
        data.resize(numel(bentuk));
        requires_grad = false;
    };

    // Constructor dengan data awal
    Tensor(const std::vector<int>& bentuk_, const std::vector<double>& data_) {
        bentuk = bentuk_;
        strides = perhitungan_strides(bentuk);
        data = data_;
        requires_grad = false;
    };

    // Copy constructor
    Tensor(const Tensor& other) 
        : data(other.data), bentuk(other.bentuk), 
          strides(other.strides), requires_grad(other.requires_grad) {};

    // Lalu kita melakukan indeksing multidimensi Tensor //

    /*
    Rumus nya tuh kek gini cuy 
    index = sigmoid besar (index x strides)
    */

    // Ini bagian Tensor terasa ajaib sih //
    int flatten_index(const std::vector<int>& indices,
                    const std::vector<int>& strides) const {
                        int index = 0;
                        for (size_t i = 0; i < indices.size(); ++i) {
                            index += indices[i] * strides[i];
                        }

                        return index;
                    };

    // Tambahin juga getter buat data dan bentuk //
    double& at(const std::vector<int>& indices) {
        return data[flatten_index(indices, strides)];
    };

    const double& at(const std::vector<int>& indices) const {
        return data[flatten_index(indices, strides)];
    };

    // Akses elemen dengan indeks flat //
    double& operator[](int i) {
        return data[i];
    };

    const double& operator[](int i) const {
        return data[i];
    };

    // OPERATOR OVERLOADING //

    // Compound assignment operators (harus jadi member functions) //

    // Operator += //
    Tensor& operator+=(const Tensor& rhs) {
        assert(bentuk == rhs.bentuk && "Shapes must match for addition");
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] += rhs.data[i];
        }
        return *this;
    };

    // Operator -= //
    Tensor& operator-=(const Tensor& rhs) {
        assert(bentuk == rhs.bentuk && "Shapes must match for subtraction");
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] -= rhs.data[i];
        }
        return *this;
    };

    // Operator *= (element-wise) //
    Tensor& operator*=(const Tensor& rhs) {
        assert(bentuk == rhs.bentuk && "Shapes must match for multiplication");
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] *= rhs.data[i];
        }
        return *this;
    };

    // Operator /= //
    Tensor& operator/=(const Tensor& rhs) {
        assert(bentuk == rhs.bentuk && "Shapes must match for division");
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] /= rhs.data[i];
        }
        return *this;
    };

    // Unary minus operator //
    Tensor operator-() const {
        Tensor result(bentuk);
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = -data[i];
        }
        return result;
    };

    // Scalar operations //
    Tensor& operator+=(double scalar) {
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] += scalar;
        }
        return *this;
    };

    Tensor& operator-=(double scalar) {
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] -= scalar;
        }
        return *this;
    };

    Tensor& operator*=(double scalar) {
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] *= scalar;
        }
        return *this;
    };

    Tensor& operator/=(double scalar) {
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] /= scalar;
        }
        return *this;
    };

    // Friend declarations for non-member operators //
    friend Tensor operator+(double s, const Tensor& t);
    friend Tensor operator-(double s, const Tensor& t);
    friend Tensor operator*(double s, const Tensor& t);
    friend Tensor operator/(double s, const Tensor& t);
    friend Tensor exp(const Tensor& t);
};

#endif 