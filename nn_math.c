#include "nn_math.h"
#include <stddef.h> // Para size_t

void dense_forward_math(const double* input, const double* weights, const double* bias,
                        double* output, int num_neurons, int input_size) {
    for (int j = 0; j < num_neurons; ++j) {
        output[j] = bias[j];
        for (int k = 0; k < input_size; ++k) {
            output[j] += input[k] * weights[j * input_size + k]; // Assumindo weights[neuronio, entrada]
        }
    }
}

void dense_backward_calc_downstream(const double* upstream_gradient, const double* weights,
                                    double* downstream_gradient, int num_neurons, int input_size) {
    for (int k = 0; k < input_size; ++k) {
        downstream_gradient[k] = 0.0;
        for (int j = 0; j < num_neurons; ++j) {
            downstream_gradient[k] += upstream_gradient[j] * weights[j * input_size + k];
        }
    }
}

void dense_backward_update_params(const double* upstream_gradient, const double* input_data,
                                  double* weights, double* biases,
                                  int num_neurons, int input_size, double learning_rate) {
    for (int j = 0; j < num_neurons; ++j) {
        for (int k = 0; k < input_size; ++k) {
            double weight_gradient = upstream_gradient[j] * input_data[k];
            weights[j * input_size + k] -= learning_rate * weight_gradient;
        }
        biases[j] -= learning_rate * upstream_gradient[j];
    }
}