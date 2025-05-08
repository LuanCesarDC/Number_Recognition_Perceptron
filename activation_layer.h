#ifndef ACTIVATION_LAYER_H_
#define ACTIVATION_LAYER_H_

#include "neural_network.h" // Para Layer

// --- Dados específicos (nenhum para Sigmoid puro) ---

// --- Função Fábrica ---
Layer *create_sigmoid_activation_layer(int size); // Tamanho = input_size = output_size

// --- Funções da Interface ---
int sigmoid_forward(Layer *layer, const double *input, double *output_buffer);
int sigmoid_backward(Layer *layer, const double *upstream_gradient,
                     const double *input_data_from_forward,
                     double *downstream_gradient_buffer, double learning_rate);
void free_activation_layer(Layer *layer);

double sigmoid(double x);
double sigmoid_deriv_from_output(double output);
int activation_save(Layer *layer, FILE *fp);

#endif // ACTIVATION_LAYER_H_