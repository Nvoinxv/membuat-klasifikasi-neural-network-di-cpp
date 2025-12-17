#ifndef DENSE_H
#define DENSE_H

#include "Tensor.h"
#include "Tensor_operator.h"
#include "Tensor_factory.h"
#include <cassert>

/*
Dense Layer (Fully Connected Layer)
Ini adalah layer yang paling dasar dalam neural network.
Dense layer melakukan operasi: output = input @ bobot + bias

Rumus Matematika:
z = X * W^T + b  (dimana X = [batch, in], W = [out, in], b = [out])
*/

class Dense {
private:
    int in_features;   // Ukuran input //
    int out_features;  // Ukuran output //
    
    Tensor bobot;    // Shape: [out_features, in_features] //
    Tensor bias;       // Shape: [out_features] //
    
    // Gradients //
    Tensor grad_bobot;
    Tensor grad_bias;
    
    // Cache untuk backward pass //
    Tensor cached_input;
    
    // Opsi untuk menggunakan bias atau tidak //
    bool gunakan_bias;

public:
    // Default constructor //
    Dense() : in_features(0), out_features(0), gunakan_bias(true) {}
    
    // Constructor dengan inisialisasi Kaiming //
    Dense(int in_features_, int out_features_, bool gunakan_bias_ = true) 
        : in_features(in_features_), out_features(out_features_), gunakan_bias(gunakan_bias_) {
        
        // Inisialisasi bobot dengan Kaiming initialization //
        // Ini optimal untuk layers yang diikuti ReLU //
        bobot = dl::kaiming_normal({out_features, in_features});
        
        // Inisialisasi bias dengan zeros //
        if (gunakan_bias) {
            bias = dl::zeros({out_features});
        }
        
        // Pre-allocate gradients //
        grad_bobot = dl::zeros({out_features, in_features});
        if (gunakan_bias) {
            grad_bias = dl::zeros({out_features});
        }
    }
    
    // Forward pass: output = input @ bobot^T + bias //
    // Input shape: [batch_size, in_features] // 
    // Output shape: [batch_size, out_features] //
    Tensor forward(const Tensor& input) {
        // Cache input untuk backward pass //
        cached_input = input;
        
        const auto& input_shape = input.get_shape();
        int batch_size = input_shape[0];
        
        // Alokasi output //
        Tensor output = dl::zeros({batch_size, out_features});
        
        // Matrix multiplication: output[i,j] = sum_k(input[i,k] * bobot[j,k]) //
        // Ini adalah X @ W^T //
        for (int b = 0; b < batch_size; ++b) {
            for (int o = 0; o < out_features; ++o) {
                double sum = 0.0;
                
                // Loop unrolling untuk in_features kecil //
                int k = 0;
                // Proses 4 elemen sekaligus //
                for (; k + 3 < in_features; k += 4) {
                    sum += input.at({b, k})     * bobot.at({o, k});
                    sum += input.at({b, k + 1}) * bobot.at({o, k + 1});
                    sum += input.at({b, k + 2}) * bobot.at({o, k + 2});
                    sum += input.at({b, k + 3}) * bobot.at({o, k + 3});
                }
                // Handle sisa elemen //
                for (; k < in_features; ++k) {
                    sum += input.at({b, k}) * bobot.at({o, k});
                }
                
                // Tambahkan bias jika ada //
                if (gunakan_bias) {
                    sum += bias[o];
                }
                
                output.at({b, o}) = sum;
            }
        }
        
        return output;
    }
    
    // Backward pass - menghitung gradients untuk bobot, bias, dan input //
    // grad_output: gradient dari loss terhadap output layer ini [batch_size, out_features] //
    // Returns: gradient terhadap input [batch_size, in_features] //
    Tensor backward(const Tensor& grad_output) {
        const auto& grad_shape = grad_output.get_shape();
        const auto& input_shape = cached_input.get_shape();
        
        int batch_size = grad_shape[0];
        
        // Reset gradients //
        grad_bobot = dl::zeros({out_features, in_features});
        if (gunakan_bias) {
            grad_bias = dl::zeros({out_features});
        }
        
        // Alokasi gradient untuk input //
        Tensor grad_input = dl::zeros({batch_size, in_features});
        
        /* 
        BACKWARD PASS:
        Kita menghitung ketiga gradient dalam satu loop untuk menghindari
        multiple passes over data dan meningkatkan cache locality.
        
        1. grad_bobot[o,i] += sum_b(grad_output[b,o] * cached_input[b,i])
        2. grad_bias[o] += sum_b(grad_output[b,o])
        3. grad_input[b,i] = sum_o(grad_output[b,o] * bobot[o,i])
        */
        
        // Single pass computation untuk bobot gradient dan bias gradient //
        for (int b = 0; b < batch_size; ++b) {
            for (int o = 0; o < out_features; ++o) {
                double grad_o = grad_output.at({b, o});
                
                // Update bias gradient (sum over batch) //
                if (gunakan_bias) {
                    grad_bias[o] += grad_o;
                }
                
                // Update weight gradients: dL/dW = X^T @ dL/dz //
                // Dengan loop unrolling untuk optimasi //
                int i = 0;
                for (; i + 3 < in_features; i += 4) {
                    grad_bobot.at({o, i})     += grad_o * cached_input.at({b, i});
                    grad_bobot.at({o, i + 1}) += grad_o * cached_input.at({b, i + 1});
                    grad_bobot.at({o, i + 2}) += grad_o * cached_input.at({b, i + 2});
                    grad_bobot.at({o, i + 3}) += grad_o * cached_input.at({b, i + 3});
                }
                for (; i < in_features; ++i) {
                    grad_bobot.at({o, i}) += grad_o * cached_input.at({b, i});
                }
            }
        }
        
        // Compute gradient untuk input: dL/dX = dL/dz @ W //
        // Ini adalah grad_output @ bobot //
        for (int b = 0; b < batch_size; ++b) {
            for (int i = 0; i < in_features; ++i) {
                double sum = 0.0;
                
                // Loop unrolling untuk out_features //
                int o = 0;
                for (; o + 3 < out_features; o += 4) {
                    sum += grad_output.at({b, o})     * bobot.at({o, i});
                    sum += grad_output.at({b, o + 1}) * bobot.at({o + 1, i});
                    sum += grad_output.at({b, o + 2}) * bobot.at({o + 2, i});
                    sum += grad_output.at({b, o + 3}) * bobot.at({o + 3, i});
                }
                for (; o < out_features; ++o) {
                    sum += grad_output.at({b, o}) * bobot.at({o, i});
                }
                
                grad_input.at({b, i}) = sum;
            }
        }
        
        return grad_input;
    }
    
    // Update bobot dengan gradient descent //
    void update_bobot(double learning_rate) {
        // bobot = bobot - learning_rate * grad_bobot //
        for (int i = 0; i < bobot.numel(); ++i) {
            bobot[i] -= learning_rate * grad_bobot[i];
        }
        
        // bias = bias - learning_rate * grad_bias
        if (gunakan_bias) {
            for (int i = 0; i < bias.numel(); ++i) {
                bias[i] -= learning_rate * grad_bias[i];
            }
        }
    }
    
    // Getters untuk bobot dan bias //
    const Tensor& dapatkan_bobot() const { return bobot; }
    const Tensor& dapatkan_bias() const { return bias; }
    
    // Getters untuk gradients //
    const Tensor& dapatkan_grad_bobot() const { return grad_bobot; }
    const Tensor& dapatkan_grad_bias() const { return grad_bias; }
    
    // Setters untuk bobot dan bias //
    void set_bobot(const Tensor& w) { bobot = w; }
    void set_bias(const Tensor& b) { bias = b; }
    
    // Info layer //
    int dapatkan_in_features() const { return in_features; }
    int dapatkan_out_features() const { return out_features; }
    bool has_bias() const { return gunakan_bias; }
    
    // Jumlah parameter //
    int num_parameters() const {
        int params = in_features * out_features;
        if (gunakan_bias) {
            params += out_features;
        }
        return params;
    }
    
    // Zero gradients - panggil sebelum training batch baru //
    void zero_grad() {
        grad_bobot = dl::zeros({out_features, in_features});
        if (gunakan_bias) {
            grad_bias = dl::zeros({out_features});
        }
    }
};

#endif