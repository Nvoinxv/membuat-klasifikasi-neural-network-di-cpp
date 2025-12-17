#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "Tensor.h"
#include "Tensor_factory.h"
#include "Tensor_operator.h"
#include "Sigmoid.h"
#include "ReLu.h"
#include "Dense.h"
#include "Adam.h"
#include "Loss.h"
#include <vector>
#include <string>
#include <iostream>

/*
Neural Network Class
Ini adalah kelas utama untuk membuat dan melatih neural network.
Cara kerja nya:
1. Tambah layer-layer (Dense, ReLU, Sigmoid)
2. Forward pass: input -> layer1 -> layer2 -> ... -> output
3. Backward pass: hitung gradient dari loss ke setiap layer
4. Update bobot dengan Adam optimizer
*/

// Enum untuk jenis layer //
enum class LayerType {
    DENSE,
    RELU,
    SIGMOID
};

// Struct untuk menyimpan informasi layer //
struct LayerInfo {
    LayerType type;
    int dense_index;  // Index ke vector dense_layers jika type == DENSE //
};

class NeuralNetwork {
private:
    std::vector<Dense> dense_layers;       // Semua Dense layers //
    std::vector<LayerInfo> layer_order;    // Urutan layer //
    std::vector<adam> optimizers;          // Adam optimizer untuk setiap Dense layer //
    
    // Cache untuk backward pass //
    std::vector<Tensor> activations;       // Menyimpan output setiap layer //
    std::vector<Tensor> pre_activations;   // Menyimpan output sebelum aktivasi (untuk ReLU backward) //
    
    double learning_rate;

public:
    // Constructor //
    NeuralNetwork(double lr = 0.001) : learning_rate(lr) {}
    
    // Factory method untuk membuat neural network //
    static NeuralNetwork membuat_neural(double learning_rate = 0.001) {
        return NeuralNetwork(learning_rate);
    }
    
    // Tambah Dense layer //
    void tambah_dense(int in_features, int out_features, bool gunakan_bias = true) {
        dense_layers.push_back(Dense(in_features, out_features, gunakan_bias));
        
        // Buat optimizer untuk layer ini //
        optimizers.push_back(adam(learning_rate));
        
        // Simpan urutan layer //
        LayerInfo info;
        info.type = LayerType::DENSE;
        info.dense_index = dense_layers.size() - 1;
        layer_order.push_back(info);
    }
    
    // Tambah ReLU activation //
    void tambah_relu() {
        LayerInfo info;
        info.type = LayerType::RELU;
        info.dense_index = -1;  // Tidak relevan untuk aktivasi //
        layer_order.push_back(info);
    }
    
    // Tambah Sigmoid activation //
    void tambah_sigmoid() {
        LayerInfo info;
        info.type = LayerType::SIGMOID;
        info.dense_index = -1;  // Tidak relevan untuk aktivasi //
        layer_order.push_back(info);
    }
    
    // FORWARD PASS //
    // Input melewati semua layer secara berurutan //
    Tensor forward(const Tensor& input) {
        // Bersihkan cache //
        activations.clear();
        pre_activations.clear();
        
        // Simpan input sebagai aktivasi pertama //
        activations.push_back(input);
        
        Tensor current = input;
        
        // Lewati setiap layer //
        for (size_t i = 0; i < layer_order.size(); ++i) {
            const LayerInfo& info = layer_order[i];
            
            switch (info.type) {
                case LayerType::DENSE: {
                    // Dense layer: output = input @ W^T + bias //
                    pre_activations.push_back(current);  // Simpan sebelum forward //
                    current = dense_layers[info.dense_index].forward(current);
                    break;
                }
                case LayerType::RELU: {
                    // ReLU: max(0, x) //
                    pre_activations.push_back(current);  // Simpan sebelum aktivasi //
                    current = ReLu::forward(current);
                    break;
                }
                case LayerType::SIGMOID: {
                    // Sigmoid: 1 / (1 + exp(-x)) //
                    pre_activations.push_back(current);  // Simpan sebelum aktivasi //
                    current = Sigmoid::forward(current);
                    break;
                }
            }
            
            // Simpan output layer ini //
            activations.push_back(current);
        }
        
        return current;
    }
    
    // BACKWARD PASS //
    // Menghitung gradient dari loss ke setiap layer //
    void backward(const Tensor& y_pred, const Tensor& y_true) {
        // Hitung gradient dari loss function (Binary Cross Entropy) //
        Tensor grad = BinaryCrossEnrtopy::backward(y_pred, y_true);
        
        // Backward melalui setiap layer (dari belakang ke depan) //
        for (int i = layer_order.size() - 1; i >= 0; --i) {
            const LayerInfo& info = layer_order[i];
            
            switch (info.type) {
                case LayerType::DENSE: {
                    // Dense backward: hitung gradient untuk bobot dan input //
                    grad = dense_layers[info.dense_index].backward(grad);
                    break;
                }
                case LayerType::RELU: {
                    // ReLU backward: grad * (1 jika x > 0, else 0) //
                    Tensor relu_grad = ReLu::backward(pre_activations[i]);
                    grad = grad * relu_grad;
                    break;
                }
                case LayerType::SIGMOID: {
                    // Sigmoid backward: grad * sigmoid(x) * (1 - sigmoid(x)) //
                    // activations[i+1] adalah output sigmoid //
                    Tensor sig_output = activations[i + 1];
                    Tensor sig_grad = Sigmoid::backward(sig_output);
                    grad = grad * sig_grad;
                    break;
                }
            }
        }
    }
    
    // OPTIMISASI (Adam) //
    // Update bobot dengan Adam optimizer //
    void optimisasi() {
        for (size_t i = 0; i < dense_layers.size(); ++i) {
            // Dapatkan bobot dan gradient dari Dense layer //
            Tensor bobot = dense_layers[i].dapatkan_bobot();
            Tensor grad_bobot = dense_layers[i].dapatkan_grad_bobot();
            Tensor bias = dense_layers[i].dapatkan_bias();
            Tensor grad_bias = dense_layers[i].dapatkan_grad_bias();
            
            // Update dengan Adam //
            optimizers[i].update(bobot, grad_bobot, bias, grad_bias);
            
            // Set kembali bobot yang sudah di-update //
            dense_layers[i].set_bobot(bobot);
            dense_layers[i].set_bias(bias);
        }
    }
    
    // TRAINING LOOP //
    // Satu langkah training lengkap //
    double train_step(const Tensor& input, const Tensor& target) {
        // 1. Zero gradients //
        zero_grad();
        
        // 2. Forward pass //
        Tensor output = forward(input);
        
        // 3. Hitung loss //
        Tensor loss_tensor = BinaryCrossEnrtopy::forward(output, target);
        double loss = 0.0;
        for (int i = 0; i < loss_tensor.numel(); ++i) {
            loss += loss_tensor[i];
        }
        loss /= loss_tensor.numel();  // Rata-rata loss //
        
        // 4. Backward pass //
        backward(output, target);
        
        // 5. Update bobot dengan Adam //
        optimisasi();
        
        return loss;
    }
    
    // Training untuk beberapa epoch //
    void train(const Tensor& X, const Tensor& y, int epochs = 100, bool verbose = true) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            double loss = train_step(X, y);
            
            if (verbose && (epoch + 1) % 10 == 0) {
                std::cout << "Epoch " << (epoch + 1) << "/" << epochs 
                          << " - Loss: " << loss << std::endl;
            }
        }
    }
    
    // Prediksi (tanpa training) //
    Tensor predict(const Tensor& input) {
        return forward(input);
    }
    
    // Zero semua gradients //
    void zero_grad() {
        for (auto& layer : dense_layers) {
            layer.zero_grad();
        }
    }
    
    // Info tentang network //
    void ringkasan() const {
        std::cout << "======== Ringkasan Nerual Network ========" << std::endl;
        std::cout << "Total layer: " << layer_order.size() << std::endl;
        std::cout << "Dense layer: " << dense_layers.size() << std::endl;
        
        int total_params = 0;
        int layer_num = 1;
        
        for (const auto& info : layer_order) {
            std::cout << "Layer " << layer_num++ << ": ";
            
            switch (info.type) {
                case LayerType::DENSE: {
                    const Dense& d = dense_layers[info.dense_index];
                    std::cout << "Dense(" << d.dapatkan_in_features() 
                              << " -> " << d.dapatkan_out_features() << ")";
                    total_params += d.num_parameters();
                    break;
                }
                case LayerType::RELU:
                    std::cout << "ReLU";
                    break;
                case LayerType::SIGMOID:
                    std::cout << "Sigmoid";
                    break;
            }
            std::cout << std::endl;
        }
        
        std::cout << "Total parameter: " << total_params << std::endl;
        std::cout << "========================================" << std::endl;
    }
};

#endif