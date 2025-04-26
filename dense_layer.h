// ======================================================
// ARQUIVO: dense_layer.h (CORRIGIDO)
// ======================================================
#ifndef DENSE_LAYER_H_
#define DENSE_LAYER_H_

#include "neural_network.h" // Para incluir a definição de Layer

// --- Estrutura de Dados Específica para Camada Densa (Atualizada) ---
typedef struct {
    int input_size;
    int num_neurons;    // output_size

    // Ponteiros para memória do Host (CPU) - SEMPRE presentes
    double* h_weights;    // <<< Renomeado de 'weights'
    double* h_biases;     // <<< Renomeado de 'biases'

#ifdef USE_CUDA // Ponteiros para memória do Device (GPU) - Opcional
    double* d_weights;
    double* d_biases;
#endif

} DenseLayerData;

// --- Função para Criar ---
Layer* create_dense_layer(int input_size, int num_neurons);

// --- Funções da Interface ---
int dense_forward(Layer* layer, const double* input, double* output_buffer);
int dense_backward(Layer* layer, const double* upstream_gradient,
                   const double* input_data_from_forward,
                   double* downstream_gradient_buffer, double learning_rate);
void free_dense_layer(Layer* layer);
int dense_save(Layer* layer, FILE* fp);

#endif // DENSE_LAYER_H_