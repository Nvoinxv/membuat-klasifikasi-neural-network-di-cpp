#ifndef ADAM_H
#define ADAM_H

#include "Tensor.h"
#include "Tensor_operator.h"
#include "Tensor_factory.h"
#include <cmath>

/*
Adam adalah metode optmisasi yang di gunakan untuk AI
untuk mengoptimalkan atau update bobot dan bias.
Jadi setelah melakukan proses forward -> backward maka 
langkah selanjut nya itu update bobot dan bias.
*/

/*
Rumus Adam seperti ini:
M = beta1 * M + (1-beta1)*grad
V = beta2 * V + (1-beta2)*grad^2

M_hat = M / (1-beta1^t)
V_hat = V / (1-beta2^t)

bobot = bobot - lr * M_hat / (sqrt(V_hat) + epsilon)
bias = bias - lr * M_hat / (sqrt(V_hat) + epsilon)
*/

class adam {
private:
    double lr;
    double beta1;
    double beta2;
    double epsilon;
    int t; // iterasi //
    
    // Ini untuk menyimpan langkah atau state //
    Tensor m_bobot, v_bobot;   // untuk bobot //
    Tensor m_bias, v_bias;     // untuk bias //
    bool inisialisasi = false;

public:
    adam(double learning_rate = 0.001, 
         double b1 = 0.9, 
         double b2 = 0.999, 
         double eps = 1e-8)
        : lr(learning_rate), beta1(b1), beta2(b2), epsilon(eps), t(0) {}
    
    void update(Tensor& bobot, Tensor& grad_bobot,
                Tensor& bias, Tensor& grad_bias) {
        // Initialize M dan V jika belum //
        if (!inisialisasi) {
            m_bobot = dl::zeros(bobot.get_shape());
            v_bobot = dl::zeros(bobot.get_shape());
            m_bias = dl::zeros(bias.get_shape());
            v_bias = dl::zeros(bias.get_shape());
            inisialisasi = true;
        }
        // Increment timestep //
        t++;
        // Update M dan V untuk bobot //
        m_bobot = beta1 * m_bobot + (1 - beta1) * grad_bobot;
        v_bobot = beta2 * v_bobot + (1 - beta2) * (grad_bobot * grad_bobot);
        // Bias correction //
        double koreksi_bias1 = 1.0 - std::pow(beta1, t);
        double koreksi_bias2 = 1.0 - std::pow(beta2, t);
        
        Tensor m_hat_bobot = m_bobot / koreksi_bias1;
        Tensor v_hat_bobot = v_bobot / koreksi_bias2;
        // Update bobot //
        bobot = bobot - lr * m_hat_bobot / (sqrt(v_hat_bobot) + epsilon);
        // Ulangi untuk bias //
        m_bias = beta1 * m_bias + (1 - beta1) * grad_bias;
        v_bias = beta2 * v_bias + (1 - beta2) * (grad_bias * grad_bias);
        
        Tensor m_hat_bias = m_bias / koreksi_bias1;
        Tensor v_hat_bias = v_bias / koreksi_bias2;
        
        bias = bias - lr * m_hat_bias / (sqrt(v_hat_bias) + epsilon);
    }
};

#endif