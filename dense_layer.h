#ifndef DENSE_LAYER_H_
#define DENSE_LAYER_H_

#include "neural_network.h" // Para incluir a definição de Layer
#include <stdio.h> // Para FILE* no save

// --- Estrutura de Dados Específica para Camada Densa (Atualizada) ---
typedef struct {
    int input_size;
    int num_neurons;    // output_size

    // Ponteiros para memória do Host (CPU) - SEMPRE presentes
    double* h_weights;
    double* h_biases;

#ifdef USE_CUDA // Ponteiros e dados para memória do Device (GPU) - Opcional
    double* d_weights;
    double* d_biases;
    double* d_activations;        // Buffer de ativações na GPU
    double* d_input_data_buffer;  // Buffer de entrada que foi usado no forward, na GPU
    double* d_downstream_gradient;// Buffer para gradiente calculado no backward, na GPU
    // Nota: O upstream_gradient vem da camada seguinte, então precisamos
    // copiar ele para a GPU a cada chamada do backward.
    double* d_temp_upstream_gradient; // Buffer temporário na GPU para o gradiente que chega
#endif

} DenseLayerData;

// --- Função para Criar ---
Layer* create_dense_layer(int input_size, int num_neurons);

// --- Funções da Interface ---
int dense_forward(Layer* layer, const double* input, double* output_buffer);
int dense_backward(Layer* layer, const double* upstream_gradient,
                   const double* input_data_from_forward, // No CPU, este é layer->input_data_buffer
                   double* downstream_gradient_buffer, double learning_rate);
void free_dense_layer(Layer* layer);
int dense_save(Layer* layer, FILE* fp);

#endif // DENSE_LAYER_H_