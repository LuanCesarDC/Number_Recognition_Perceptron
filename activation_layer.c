#include "activation_layer.h"
#include <math.h> // Para exp
#include <stdio.h>
#include <stdlib.h>
#include <string.h> // Para memcpy

// Função Sigmoid e sua derivada (usando a saída)
double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
double sigmoid_deriv_from_output(double output) { return output * (1.0 - output); }

// --- Implementações da Interface ---

int sigmoid_forward(Layer *layer, const double *input, double *output_buffer)
{
    if (!layer || !input || !output_buffer)
        return -1;

    // Aplica sigmoid elemento a elemento
    for (int i = 0; i < layer->input_size; ++i)
    { // input_size == output_size
        output_buffer[i] = sigmoid(input[i]);
    }

    // Salva a saída no buffer de ativações da Layer genérica
    memcpy(layer->activations, output_buffer, layer->output_size * sizeof(double));
    // Salva a entrada (soma ponderada da camada anterior) para o backward
    memcpy(layer->input_data_buffer, input, layer->input_size * sizeof(double));

    return 0;
}

int sigmoid_backward(Layer *layer, const double *upstream_gradient,
                     const double *input_data_from_forward, // Usado para pegar a ativação (saída do forward)
                     double *downstream_gradient_buffer, double learning_rate)
{
    if (!layer || !upstream_gradient || !downstream_gradient_buffer || !layer->activations)
        return -1;

    // Calcula o gradiente para a camada anterior:
    // downstream_grad = upstream_grad * sigmoid'(soma_ponderada_que_entrou)
    //                 = upstream_grad * sigmoid(soma) * (1 - sigmoid(soma))
    //                 = upstream_grad * activation * (1 - activation)
    for (int i = 0; i < layer->input_size; ++i)
    {                                              // input_size == output_size
        double activation = layer->activations[i]; // Pega a ativação calculada no forward
        double deriv = sigmoid_deriv_from_output(activation);
        downstream_gradient_buffer[i] = upstream_gradient[i] * deriv;
    }
    // Salvar este gradiente calculado no buffer da Layer genérica
    memcpy(layer->downstream_gradient, downstream_gradient_buffer, layer->input_size * sizeof(double));

    return 0;
}

int activation_save(Layer* layer, FILE* fp) {
    // Verifica ponteiros básicos, mas não há o que escrever
    if (!layer || !fp) return -1;
    // Não faz nada, pois não há pesos ou biases aqui.
    return 0; // Sucesso
}

void free_activation_layer(Layer *layer)
{
    if (!layer)
        return;
    // Não há dados específicos (internal_data) para liberar para Sigmoid
    // Libera buffers da Layer genérica
    free(layer->activations);
    free(layer->input_data_buffer);
    free(layer->downstream_gradient);
    // Libera a própria struct Layer genérica
    free(layer);
}

// --- Função Fábrica ---
Layer *create_sigmoid_activation_layer(int size)
{
    // 1. Alocar a Layer genérica
    Layer *layer = (Layer *)malloc(sizeof(Layer));
    if (!layer)
        return NULL;

    // 2. Alocar buffers na Layer genérica
    layer->activations = (double *)malloc(size * sizeof(double));
    layer->input_data_buffer = (double *)malloc(size * sizeof(double));
    layer->downstream_gradient = (double *)malloc(size * sizeof(double)); // Gradiente que sai

    if (!layer->activations || !layer->input_data_buffer || !layer->downstream_gradient)
    {
        free(layer->activations);
        free(layer->input_data_buffer);
        free(layer->downstream_gradient);
        free(layer);
        return NULL;
    }

    // 3. Configurar a Layer genérica
    layer->internal_data = NULL; // Sem dados específicos
    layer->input_size = size;
    layer->output_size = size;
    layer->forward = sigmoid_forward;
    layer->backward = sigmoid_backward;
    layer->free_layer = free_activation_layer;
    layer->save = activation_save;

    return layer;
}