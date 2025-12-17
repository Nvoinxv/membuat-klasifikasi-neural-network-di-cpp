#include "include/Tensor.h"
#include "include/Tensor_factory.h"
#include "include/Sigmoid.h"
#include "include/ReLu.h"
#include "include/Dense.h"
#include "include/Adam.h"
#include "include/Loss.h"
#include "include/NeuralNetwork.h"

// Ini buat menjalankan program nya disini //
int main() {
    // Hanya dummy data ya bukan real data //
    // Karna gw rada males pakai real data //

    // 1D Tensor//
    auto a = dl::tensor({1.0, 2.0, 3.0, 4.0, 5.0});
    std::cout << "1D Tensor: " << a << std::endl;
    std::cout << std::endl;

    // 2D Tensor //
    auto b = dl::tensor({
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    });
    std::cout << "2D Tensor: " << b << std::endl;
    std::cout << std::endl;

    // Fungsi faktori  //

    // Zeros - tensor isi 0 //
    auto zeros = dl::zeros({2, 3});
    std::cout << "Zeros: " << zeros << std::endl;
    std::cout << std::endl;

    // Ones - tensor isi 1 //
    auto ones = dl::ones({3, 2});
    std::cout << "Ones: " << ones << std::endl;
    std::cout << std::endl;

    // Random Normal - untuk inisialisasi bobot //
    dl::manual_seed(42);  // Set seed untuk reproducibility //
    auto random = dl::randn({2, 3});
    std::cout << "Random Normal: " << random << std::endl;
    std::cout << std::endl;

    // fungsi urutan //
    // Arange //
    auto seq = dl::arange(0, 10, 2);
    std::cout << "Arange: " << seq << std::endl;
    std::cout << std::endl;

    // Linspace - nilai linear //
    auto lin = dl::linspace(0, 1, 5);
    std::cout << "Linspace: " << lin << std::endl;
    std::cout << std::endl;

    // Special Tensors //
    // Identity Matrix //
    auto identity = dl::eye(3);
    std::cout << "Identity 3x3: " << identity << std::endl;
    std::cout << std::endl;

    // Weight Initialization  //
    
    // Xavier init - bagus untuk sigmoid //
    auto xavier_w = dl::xavier_uniform({64, 128});
    std::cout << "Xavier Uniform (64x128): " << xavier_w << std::endl;
    std::cout << std::endl;

    // Kaiming init - bagus untuk ReLU //
    auto kaiming_w = dl::kaiming_normal({128, 64});
    std::cout << "Kaiming Normal (128x64): " << kaiming_w << std::endl;
    std::cout << std::endl;

    // Tensor Operations  //
    auto x = dl::tensor({1.0, 2.0, 3.0});
    auto y = dl::tensor({4.0, 5.0, 6.0});
    
    std::cout << "x: " << x << std::endl;
    std::cout << "y: " << y << std::endl;
    std::cout << "x + y: " << (x + y) << std::endl;
    std::cout << "x * y: " << (x * y) << std::endl;
    std::cout << "x * 2.0: " << (x * 2.0) << std::endl;
    std::cout << std::endl;
    
    // Testing bagian aktivasi, apakah sudah works atau belum //
    // Ini hanya gw ambil bagian forward saja bukan backward //
    // Sigmoid Forward //
    std::cout << "SIGMOID TEST" << std::endl;
    auto input = dl::tensor({{-1.0, 0.0, 1.0}, {2.0, -2.0, 0.5}});
    std::cout << "Input Sigmoid: " << input << std::endl;
    
    auto sigmoid_out = Sigmoid::forward(input);
    std::cout << "Output Sigmoid: " << sigmoid_out << std::endl;

    // ReLu Forward //
    std::cout << "ReLu TEST" << std::endl;
    auto input_relu = dl::tensor({{-1.0, 0.0, 1.0}, {3.5, -10.0, 0.0}});
    std::cout << "Input ReLu: " << input_relu << std::endl;

    auto relu_out = ReLu::forward(input_relu);
    std::cout << "Output ReLu: " << relu_out << std::endl;

    // Testing bagian neural Netwrok //
    // Ini gw isi learning rate dan anggap saja ini 0.001 misalkan //
    NeuralNetwork nn = NeuralNetwork::membuat_neural(0.001);
    
    // Tambah layers //
    // Ini input nya baru dummy btw //
    // Gw mau buat flow contoh nya 2 input -> 4 hidden layers -> 1 output //
    // Dan gw ada tambahan relu di tengah setelah dense dan sigmoid di akhir //
    nn.tambah_dense(2,4); 
    nn.tambah_dense(4,1);
    nn.tambah_relu();
    nn.tambah_sigmoid();

    // Lihat ringkasan //
    nn.ringkasan();

    // Buat data training dummy nya //
    // Ini bagian feature atau x variabel //
    auto X = dl::tensor({
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    });

    // Ini bagian target atau y variabel //
    // Disini gw emang fokus nya supervised learning //
    // Karna kebanyakan deep learning lebih supervised //
    auto y_target = dl::tensor({4, 1}, {0.0, 1.0, 1.0, 0.0}); // Ini gw isi shape nya ya btw //

    // Jadi ukuran shape dan data harus sama dari Y dan X nya //
    // Disini gw pakai 4 shape atau bentuk dan 1 data //

    // Training atau pelatihan si AI  nya //
    // Gw pakai 100 epoch //
    nn.train(X, y_target, 100, true);

    // Lalu lakukan prediksi setelah pelatihan //
    auto output = nn.predict(X);
    std::cout << "Prediksi: " << output << std::endl;

    return 0;
}