#include "dense_layer.h" // Inclui a definição CORRETA de DenseLayerData
#include "nn_math.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef USE_CUDA
#include <cuda_runtime.h>
// Definição da macro de check (simplificada, idealmente de um header comum)
#define CUDA_CHECK(call) do { /* ... (definição completa da macro) ... */ } while(0)
#endif

static void initialize_weights(double* weights, int rows, int cols);


// Função auxiliar para inicializar pesos (definição)
static void initialize_weights(double* weights, int rows, int cols) { // Adicionar 'static'
    for (int i = 0; i < rows * cols; ++i) {
        weights[i] = ((double)rand() / RAND_MAX) * 0.2 - 0.1;
    }
}

int dense_forward(Layer* layer, const double* input, double* output_buffer) {
    // ... (Código como na resposta anterior, usando data->h_weights/h_biases para CPU) ...
    // ... (e data->d_weights/d_biases para GPU) ...
#ifdef USE_CUDA
    // ... Código CUDA ...
     DenseLayerData* data = (DenseLayerData*)layer->internal_data;
    // Exemplo de chamada (assumindo ponteiros GPU corretos para input/output):
    // dense_forward_math(d_input, data->d_weights, data->d_biases, d_output_buffer, ...);
#else
    // --- Versão CPU ---
     DenseLayerData* data = (DenseLayerData*)layer->internal_data;
    memcpy(layer->input_data_buffer, input, layer->input_size * sizeof(double));
    dense_forward_math(input, data->h_weights, data->h_biases, output_buffer, // OK agora
                       data->num_neurons, data->input_size);
    memcpy(layer->activations, output_buffer, layer->output_size * sizeof(double));
#endif
    return 0;
}

int dense_backward(Layer* layer, const double* upstream_gradient,
                   const double* input_data_from_forward,
                   double* downstream_gradient_buffer, double learning_rate) {
    // ... (Código como na resposta anterior, usando h_weights/h_biases para CPU) ...
    // ... (e d_weights/d_biases para GPU) ...
#ifdef USE_CUDA
    // ... Código CUDA ...
     DenseLayerData* data = (DenseLayerData*)layer->internal_data;
    // Exemplo de chamadas (assumindo ponteiros GPU corretos):
    // dense_backward_calc_downstream(d_upstream_gradient, data->d_weights, ...);
    // dense_backward_update_params(d_upstream_gradient, d_input_data, data->d_weights, data->d_biases, ...);
#else
    // --- Versão CPU ---
    DenseLayerData* data = (DenseLayerData*)layer->internal_data;
    dense_backward_calc_downstream(upstream_gradient, data->h_weights, // OK agora
                                   downstream_gradient_buffer,
                                   data->num_neurons, data->input_size);
     memcpy(layer->downstream_gradient, downstream_gradient_buffer, layer->input_size * sizeof(double));
    dense_backward_update_params(upstream_gradient, input_data_from_forward,
                                 data->h_weights, data->h_biases, // OK agora
                                 data->num_neurons, data->input_size, learning_rate);
#endif
    return 0;
}

int dense_save(Layer* layer, FILE* fp) {
    // ... (Código como na resposta anterior, usando h_weights/h_biases para salvar) ...
     if (!layer || !layer->internal_data || !fp) return -1;
    DenseLayerData* data = (DenseLayerData*)layer->internal_data;
#ifdef USE_CUDA
    // ... (código para copiar Device->Host antes de salvar) ...
#endif
    // Escrever dados do HOST (h_weights, h_biases)
    if (fwrite(data->h_biases, sizeof(double), data->num_neurons, fp) != data->num_neurons) { /* erro */ return -1; } // OK agora
    size_t num_weights = (size_t)data->num_neurons * data->input_size;
    if (fwrite(data->h_weights, sizeof(double), num_weights, fp) != num_weights) { /* erro */ return -1; } // OK agora
    return 0;
}

void free_dense_layer(Layer* layer) {
    // ... (Código como na resposta anterior, usando h_weights/h_biases e d_weights/d_biases) ...
     if (!layer) return;
    if (layer->internal_data) {
        DenseLayerData* data = (DenseLayerData*)layer->internal_data;
        free(data->h_weights); // OK agora
        free(data->h_biases);  // OK agora
#ifdef USE_CUDA
        cudaFree(data->d_weights);
        cudaFree(data->d_biases);
#endif
        free(data);
    }
    free(layer->activations);
    free(layer->input_data_buffer);
    free(layer->downstream_gradient);
    free(layer);
}


// --- Função Fábrica (Deve funcionar como na resposta anterior agora) ---
Layer* create_dense_layer(int input_size, int num_neurons) {
     // ... (Código como na resposta anterior, alocando/usando h_weights/h_biases) ...
     // ... (e d_weights/d_biases se USE_CUDA) ...
    DenseLayerData* data = (DenseLayerData*)malloc(sizeof(DenseLayerData));
    Layer* layer = (Layer*)malloc(sizeof(Layer));
    if (!data || !layer) { /* erro free */ return NULL; }

    data->input_size = input_size;
    data->num_neurons = num_neurons;
    size_t weights_size = (size_t)num_neurons * input_size * sizeof(double);
    size_t biases_size = num_neurons * sizeof(double);

    // Alocar memória HOST
    data->h_weights = (double*)malloc(weights_size); // OK agora
    data->h_biases = (double*)malloc(biases_size);   // OK agora

    // Alocar buffers genéricos no HOST
    layer->activations = (double*)malloc(num_neurons * sizeof(double));
    layer->input_data_buffer = (double*)malloc(input_size * sizeof(double));
    layer->downstream_gradient = (double*)malloc(input_size * sizeof(double));

    if (!data->h_weights || !data->h_biases || !layer->activations || !layer->input_data_buffer || !layer->downstream_gradient) {
         // ... liberar tudo o que foi alocado ...
         return NULL;
    }

    // Inicializar pesos/biases no HOST
    initialize_weights(data->h_weights, num_neurons, input_size); // OK agora
    for (int i = 0; i < num_neurons; ++i) data->h_biases[i] = 0.0; // OK agora

#ifdef USE_CUDA
    // Alocar memória DEVICE
    CUDA_CHECK(cudaMalloc((void**)&data->d_weights, weights_size));
    CUDA_CHECK(cudaMalloc((void**)&data->d_biases, biases_size));
    // Copiar pesos/biases inicializados do Host para Device
    CUDA_CHECK(cudaMemcpy(data->d_weights, data->h_weights, weights_size, cudaMemcpyHostToDevice)); // OK agora
    CUDA_CHECK(cudaMemcpy(data->d_biases, data->h_biases, biases_size, cudaMemcpyHostToDevice)); // OK agora
#endif

    // Configurar Layer genérica
    layer->internal_data = data;
    layer->input_size = input_size;
    layer->output_size = num_neurons;
    layer->forward = dense_forward;
    layer->backward = dense_backward;
    layer->free_layer = free_dense_layer;
    layer->save = dense_save;

    return layer;
}