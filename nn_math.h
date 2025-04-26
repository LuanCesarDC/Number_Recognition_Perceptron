#ifndef NN_MATH_H_
#define NN_MATH_H_

// Estrutura para representar vetores/matrizes (simplificado)
// Poderíamos ter structs mais elaboradas, mas vamos usar ponteiros e tamanhos
// por enquanto, assumindo layout linear (row-major ou col-major).

// Forward: Calcula output = (input * weights^T) + bias
// input: (1 x input_size)
// weights: (num_neurons x input_size)
// bias: (num_neurons x 1)
// output: (num_neurons x 1) - Armazena o resultado
void dense_forward_math(const double* input, const double* weights, const double* bias,
                        double* output, int num_neurons, int input_size);

// Backward (Cálculo do gradiente para camada anterior):
// downstream_gradient = upstream_gradient * weights
// upstream_gradient: (1 x num_neurons)
// weights: (num_neurons x input_size)
// downstream_gradient: (1 x input_size) - Armazena o resultado
void dense_backward_calc_downstream(const double* upstream_gradient, const double* weights,
                                    double* downstream_gradient, int num_neurons, int input_size);

// Backward (Atualização de pesos e biases):
// grad_weights = upstream_gradient^T * input_data
// grad_bias = upstream_gradient
// weights -= learning_rate * grad_weights
// biases -= learning_rate * grad_bias
// upstream_gradient: (1 x num_neurons)
// input_data: (1 x input_size)
// weights: (num_neurons x input_size) - Será modificado
// biases: (num_neurons x 1) - Será modificado
void dense_backward_update_params(const double* upstream_gradient, const double* input_data,
                                  double* weights, double* biases,
                                  int num_neurons, int input_size, double learning_rate);

#endif // NN_MATH_H_