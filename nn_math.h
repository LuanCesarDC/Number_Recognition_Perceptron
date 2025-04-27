#ifndef NN_MATH_H_
#define NN_MATH_H_

// Estrutura para representar vetores/matrizes (simplificado)
// ... (comentários como antes) ...

// Garante que compiladores C++ entendam que estas são funções C
#ifdef __cplusplus
extern "C" {
#endif

// === Declarações das Funções ===

// Forward: Calcula output = (input * weights^T) + bias
void dense_forward_math(const double* input, const double* weights, const double* bias,
                        double* output, int num_neurons, int input_size);

// Backward (Cálculo do gradiente para camada anterior):
// downstream_gradient = upstream_gradient * weights
void dense_backward_calc_downstream(const double* upstream_gradient, const double* weights,
                                    double* downstream_gradient, int num_neurons, int input_size);

// Backward (Atualização de pesos e biases):
void dense_backward_update_params(const double* upstream_gradient, const double* input_data,
                                  double* weights, double* biases,
                                  int num_neurons, int input_size, double learning_rate);


#ifdef __cplusplus
} // extern "C"
#endif

#endif // NN_MATH_H_