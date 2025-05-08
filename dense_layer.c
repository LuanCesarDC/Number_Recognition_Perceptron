#include "dense_layer.h"
#include "nn_math.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h> 

#ifdef USE_CUDA
#include <cuda_runtime.h>
// Definição da macro de check CUDA (completa)
#define CUDA_CHECK(call)                                                                                           \
    do                                                                                                             \
    {                                                                                                              \
        cudaError_t err = call;                                                                                    \
        if (err != cudaSuccess)                                                                                    \
        {                                                                                                          \
            fprintf(stderr, "CUDA Error in %s:%d - %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                                                    \
        }                                                                                                          \
    } while (0)
#endif

// Função auxiliar para inicializar pesos (definição)
static void initialize_weights(double* weights, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        // Inicialização Xavier/Glorot
         double range = sqrt(6.0 / (rows + cols)); // sqrt agora é conhecido
         weights[i] = ((double)rand() / RAND_MAX) * 2.0 * range - range;
    }
}

// --- Implementações da Interface ---

int dense_forward(Layer* layer, const double* input, double* output_buffer) {
    if (!layer || !layer->internal_data || !input || !output_buffer) return -1;
    DenseLayerData* data = (DenseLayerData*)layer->internal_data;

#ifdef USE_CUDA
    // --- Versão CUDA ---
    size_t input_bytes = layer->input_size * sizeof(double);
    size_t output_bytes = layer->output_size * sizeof(double);

    // 1. Copiar input do Host (CPU) para o buffer de input da GPU
    CUDA_CHECK(cudaMemcpy(data->d_input_data_buffer, input, input_bytes, cudaMemcpyHostToDevice));

    // 2. Chamar a função matemática da GPU (agora lança o kernel manual)
    //    Usa: d_input_buffer, d_weights, d_biases. Escreve em: d_activations
    dense_forward_math(data->d_input_data_buffer, data->d_weights, data->d_biases,
                       data->d_activations, data->num_neurons, data->input_size);

    // 3. Copiar resultado (ativações) da GPU para o output_buffer do Host (CPU)
    CUDA_CHECK(cudaMemcpy(output_buffer, data->d_activations, output_bytes, cudaMemcpyDeviceToHost));

    // 4. Copiar também para o buffer de ativações genérico da camada no Host (redundante?)
    //    Geralmente a próxima camada pegará 'output_buffer' como entrada.
    //    Mas se precisarmos das ativações no Host depois, fazemos a cópia.
    memcpy(layer->activations, output_buffer, output_bytes);

    // 5. Salvar a ENTRADA ORIGINAL do HOST no buffer genérico input_data_buffer da camada no HOST
    //    Isto é necessário se a função backward da *próxima* camada (e.g., ativação)
    //    precisar da entrada que *esta* camada recebeu (antes da transformação).
    //    No entanto, a *nossa* função dense_backward precisa da entrada que foi usada
    //    no cálculo da GPU, que já está em data->d_input_data_buffer.
    memcpy(layer->input_data_buffer, input, input_bytes);


#else
    // --- Versão CPU ---
    // Salva a entrada no buffer genérico (para backward)
    memcpy(layer->input_data_buffer, input, layer->input_size * sizeof(double));
    // Calcula a saída
    dense_forward_math(input, data->h_weights, data->h_biases, output_buffer,
                       data->num_neurons, data->input_size);
    // Salva a saída no buffer de ativações genérico
    memcpy(layer->activations, output_buffer, layer->output_size * sizeof(double));
#endif
    return 0;
}

int dense_backward(Layer* layer, const double* upstream_gradient, // Vem da camada seguinte, está no HOST
                   const double* input_data_from_forward, // Vem do buffer da camada atual (HOST)
                   double* downstream_gradient_buffer,   // Onde escrever o resultado (HOST)
                   double learning_rate) {
    if (!layer || !layer->internal_data || !upstream_gradient || !input_data_from_forward || !downstream_gradient_buffer) return -1;
    DenseLayerData* data = (DenseLayerData*)layer->internal_data;

#ifdef USE_CUDA
    // --- Versão CUDA ---
    size_t upstream_bytes = layer->output_size * sizeof(double); // output_size == num_neurons
    size_t downstream_bytes = layer->input_size * sizeof(double);

    // 1. Copiar upstream_gradient (do Host) para buffer temporário na GPU
    CUDA_CHECK(cudaMemcpy(data->d_temp_upstream_gradient, upstream_gradient, upstream_bytes, cudaMemcpyHostToDevice));

    // 2. Calcular gradiente para a camada anterior (downstream) na GPU
    //    Usa: d_temp_upstream, d_weights. Escreve em: d_downstream_gradient
    dense_backward_calc_downstream(data->d_temp_upstream_gradient, data->d_weights,
                                   data->d_downstream_gradient,
                                   data->num_neurons, data->input_size);

    // 3. Copiar downstream_gradient calculado da GPU para o buffer do Host
    CUDA_CHECK(cudaMemcpy(downstream_gradient_buffer, data->d_downstream_gradient, downstream_bytes, cudaMemcpyDeviceToHost));
    //    Copiar também para o buffer genérico da camada no Host
    memcpy(layer->downstream_gradient, downstream_gradient_buffer, downstream_bytes);

    // 4. Atualizar pesos e biases na GPU
    //    Usa: d_temp_upstream, d_input_data_buffer (que foi salvo no forward), d_weights, d_biases
    //    Modifica: d_weights, d_biases
    dense_backward_update_params(data->d_temp_upstream_gradient, data->d_input_data_buffer,
                                 data->d_weights, data->d_biases,
                                 data->num_neurons, data->input_size, learning_rate);

#else
    // --- Versão CPU ---
    // Calcular gradiente para a camada anterior
    dense_backward_calc_downstream(upstream_gradient, data->h_weights,
                                   downstream_gradient_buffer,
                                   data->num_neurons, data->input_size);
    // Salvar no buffer genérico da camada
     memcpy(layer->downstream_gradient, downstream_gradient_buffer, layer->input_size * sizeof(double));

    // Atualizar pesos e biases (usa input_data_from_forward que é layer->input_data_buffer)
    dense_backward_update_params(upstream_gradient, input_data_from_forward,
                                 data->h_weights, data->h_biases,
                                 data->num_neurons, data->input_size, learning_rate);
#endif
    return 0;
}

int dense_save(Layer* layer, FILE* fp) {
    if (!layer || !layer->internal_data || !fp) return -1;
   DenseLayerData* data = (DenseLayerData*)layer->internal_data;

#ifdef USE_CUDA
   // --- Versão CUDA: Copiar parâmetros da GPU para a CPU ANTES de salvar ---
   // Mover declarações para dentro do #ifdef
   size_t weights_bytes = (size_t)data->num_neurons * data->input_size * sizeof(double); // <<< MOVIDO AQUI
   size_t biases_bytes = data->num_neurons * sizeof(double); // <<< MOVIDO AQUI
   printf("CUDA Save: Copiando pesos/biases D->H... ");
   CUDA_CHECK(cudaMemcpy(data->h_weights, data->d_weights, weights_bytes, cudaMemcpyDeviceToHost));
   CUDA_CHECK(cudaMemcpy(data->h_biases, data->d_biases, biases_bytes, cudaMemcpyDeviceToHost));
   printf("OK.\n");
#endif

   // Escrever dados do HOST (h_weights, h_biases)
   // Corrigir comparação de fwrite
   if (fwrite(data->h_biases, sizeof(double), data->num_neurons, fp) != (size_t)data->num_neurons) { // <<< CAST ADICIONADO
       perror("Erro ao escrever biases");
       return -1;
   }
   // Calcular num_weights diretamente aqui para o lado não-CUDA
   size_t num_weights_to_write = (size_t)data->num_neurons * data->input_size;
   if (fwrite(data->h_weights, sizeof(double), num_weights_to_write, fp) != num_weights_to_write) {
        perror("Erro ao escrever weights");
       return -1;
   }
   return 0;
}

void free_dense_layer(Layer* layer) {
     if (!layer) return;

    // Liberar buffers genéricos do Host (sempre existem)
    free(layer->activations);
    free(layer->input_data_buffer);
    free(layer->downstream_gradient);

    // Liberar dados específicos da camada densa
    if (layer->internal_data) {
        DenseLayerData* data = (DenseLayerData*)layer->internal_data;

        // Liberar memória do Host
        free(data->h_weights);
        free(data->h_biases);

#ifdef USE_CUDA
        // Liberar memória do Device
        cudaFree(data->d_weights);
        cudaFree(data->d_biases);
        cudaFree(data->d_activations);
        cudaFree(data->d_input_data_buffer);
        cudaFree(data->d_downstream_gradient);
        cudaFree(data->d_temp_upstream_gradient);
#endif
        // Liberar a struct de dados específicos
        free(data);
    }
    // Liberar a struct Layer genérica
    free(layer);
}


// --- Função Fábrica ---
Layer* create_dense_layer(int input_size, int num_neurons) {
    DenseLayerData* data = NULL;
    Layer* layer = NULL;
    size_t weights_bytes = (size_t)num_neurons * input_size * sizeof(double);
    size_t biases_bytes = (size_t)num_neurons * sizeof(double);
    size_t input_bytes = (size_t)input_size * sizeof(double);
    size_t output_bytes = (size_t)num_neurons * sizeof(double);
    // int error = 0; // <<< REMOVIDO

    // Alocar structs principais
    data = (DenseLayerData*)malloc(sizeof(DenseLayerData));
    layer = (Layer*)malloc(sizeof(Layer));
    if (!data || !layer) { goto cleanup; } // <<< Simplificado
    memset(data, 0, sizeof(DenseLayerData));
    memset(layer, 0, sizeof(Layer));

    data->input_size = input_size;
    data->num_neurons = num_neurons;

    // Alocar memória HOST para pesos, biases e buffers genéricos
    data->h_weights = (double*)malloc(weights_bytes);
    data->h_biases = (double*)malloc(biases_bytes);
    layer->activations = (double*)malloc(output_bytes);
    layer->input_data_buffer = (double*)malloc(input_bytes);
    layer->downstream_gradient = (double*)malloc(input_bytes);

    if (!data->h_weights || !data->h_biases || !layer->activations || !layer->input_data_buffer || !layer->downstream_gradient) {
        goto cleanup; // <<< Simplificado
    }

    // Inicializar pesos/biases no HOST
    initialize_weights(data->h_weights, num_neurons, input_size);
    for (int i = 0; i < num_neurons; ++i) data->h_biases[i] = 0.0;

#ifdef USE_CUDA
    // ... (Alocação CUDA como antes) ...
    CUDA_CHECK(cudaMalloc((void**)&data->d_weights, weights_bytes));
    CUDA_CHECK(cudaMalloc((void**)&data->d_biases, biases_bytes));
    CUDA_CHECK(cudaMalloc((void**)&data->d_activations, output_bytes));
    CUDA_CHECK(cudaMalloc((void**)&data->d_input_data_buffer, input_bytes));
    CUDA_CHECK(cudaMalloc((void**)&data->d_downstream_gradient, input_bytes));
    CUDA_CHECK(cudaMalloc((void**)&data->d_temp_upstream_gradient, output_bytes));

    CUDA_CHECK(cudaMemcpy(data->d_weights, data->h_weights, weights_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data->d_biases, data->h_biases, biases_bytes, cudaMemcpyHostToDevice));
#endif

    // ... (Configurar Layer genérica como antes) ...
    layer->internal_data = data;
    layer->input_size = input_size;
    layer->output_size = num_neurons;
    layer->forward = dense_forward;
    layer->backward = dense_backward;
    layer->free_layer = free_dense_layer;
    layer->save = dense_save;

    return layer;

cleanup: // Label permanece o mesmo
    fprintf(stderr, "Falha ao criar camada densa - limpando memória parcial.\n");
    // ... (lógica de cleanup como antes) ...
     if (layer) {
        free(layer->activations);
        free(layer->input_data_buffer);
        free(layer->downstream_gradient);
    }
    if (data) {
        free(data->h_weights);
        free(data->h_biases);
#ifdef USE_CUDA
        if(data->d_weights) cudaFree(data->d_weights);
        if(data->d_biases) cudaFree(data->d_biases);
        if(data->d_activations) cudaFree(data->d_activations);
        if(data->d_input_data_buffer) cudaFree(data->d_input_data_buffer);
        if(data->d_downstream_gradient) cudaFree(data->d_downstream_gradient);
        if(data->d_temp_upstream_gradient) cudaFree(data->d_temp_upstream_gradient);
#endif
        free(data);
    }
    free(layer);
    return NULL;
}