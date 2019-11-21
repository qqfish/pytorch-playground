#include <torch/torch.h>
#include <iostream>
#include <ctime>

torch::Tensor ComputeMnist(const torch::Tensor& w1,
                  const torch::Tensor& b1, const torch::Tensor& w2,
                  const torch::Tensor& b2,
                  const torch::Tensor& test_input_feature,
                  const torch::Tensor& test_input_labels) {
  torch::Tensor a1 = test_input_feature.matmul(w1);
  torch::Tensor z1 = a1.add(b1);
  torch::Tensor activation1 = z1.relu();
  torch::Tensor a2 = activation1.matmul(w2);
  torch::Tensor z2 = a2.add(b2);
  torch::Tensor argmax_h2 = z2.argmax();
  torch::Tensor equal_i32 = argmax_h2.eq(test_input_labels);
  torch::Tensor equal_f32 = equal_i32.to(torch::kFloat32);
  torch::Tensor avg_accuracy = equal_f32.mean(0);
  return avg_accuracy;
}

int main() {
  torch::Tensor w1 = torch::ones({784, 512}, torch::kFloat32);
  torch::Tensor b1 = torch::ones({512}, torch::kFloat32);
  torch::Tensor w2 = torch::ones({512, 10}, torch::kFloat32);
  torch::Tensor b2 = torch::ones({10}, torch::kFloat32);
  torch::Tensor test_input_feature = torch::ones({100, 784}, torch::kFloat32);
  torch::Tensor test_input_labels = torch::ones({100}, torch::kInt32);

  int iters = 1000;
  clock_t begin = clock();
  for (int i = 0; i < iters; i++) {
    auto avg_accuracy = ComputeMnist(w1, b1, w2, b2, test_input_feature, test_input_labels);
  }
  clock_t end = clock();
  double elapsed_us = double(end - begin) / CLOCKS_PER_SEC * 1000000;
  std::cout << iters << " steps per-step : "<< elapsed_us / iters << "us" << std::endl;
}
