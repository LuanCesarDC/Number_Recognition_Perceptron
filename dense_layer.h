#ifndef DENSE_LAYER_H_
#define DENSE_LAYER_H_

#include "neural_network.h" // Para incluir a definição de Layer

// --- Estrutura de Dados Específica para Camada Densa ---
typedef struct
{
    int input_size;
    int num_neurons; // output_size
    double *weights; // Matriz: num_neurons x input_size
    double *biases;  // Vetor: num_neurons
    // Não precisa de 'activations', 'errors' aqui, pois eles estarão na Layer genérica
} DenseLayerData;

// --- Função para Criar a Camada Densa ---
// Retorna um ponteiro para a Layer genérica configurada, ou NULL em erro.
Layer *create_dense_layer(int input_size, int num_neurons);

// --- Funções que implementam a interface Layer (serão usadas pelos ponteiros) ---
// Estas não precisam ser expostas no .h se apenas create_dense_layer for público,
// mas vamos deixá-las para clareza.
int dense_forward(Layer *layer, const double *input, double *output_buffer);
int dense_backward(Layer *layer, const double *upstream_gradient,
                   const double *input_data_from_forward,
                   double *downstream_gradient_buffer, double learning_rate);
void free_dense_layer(Layer *layer);
int dense_save(Layer* layer, FILE* fp);

#endif // DENSE_LAYER_H_