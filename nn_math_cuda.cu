#include "nn_math.h"
#include <cuda_runtime.h>
#include <stdio.h> // Para error checking
#include <math.h> // Para ceil

// Macro para checagem de erro CUDA (essencial!)
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

// --- Kernels CUDA ---

// Kernel para: output = (input * weights^T) + bias
__global__ void dense_forward_kernel(const double* input, const double* weights, const double* bias,
                                     double* output, int num_neurons, int input_size)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Índice do neurônio de saída

    if (j < num_neurons) {
        double sum = bias[j]; // Inicializa com o bias
        for (int k = 0; k < input_size; ++k) {
            // weights está em layout row-major: weights[neuronio, entrada]
            sum += input[k] * weights[j * input_size + k];
        }
        output[j] = sum;
    }
}

// Kernel para: downstream_gradient = upstream_gradient * weights
// (Equivalente a downstream = weights^T * upstream)
__global__ void dense_backward_calc_downstream_kernel(const double* upstream_gradient, const double* weights,
                                                      double* downstream_gradient, int num_neurons, int input_size)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x; // Índice do gradiente de saída (para camada anterior)

    if (k < input_size) {
        double sum = 0.0;
        for (int j = 0; j < num_neurons; ++j) {
            // weights[neuronio, entrada]
            sum += upstream_gradient[j] * weights[j * input_size + k];
        }
        downstream_gradient[k] = sum;
    }
}

// Kernel para: biases -= learning_rate * upstream_gradient
__global__ void dense_backward_update_biases_kernel(const double* upstream_gradient, double* biases,
                                                    int num_neurons, double learning_rate)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Índice do bias/neurônio

    if (j < num_neurons) {
        biases[j] -= learning_rate * upstream_gradient[j];
    }
}

// Kernel para: weights -= learning_rate * (upstream_gradient^T * input_data)
// (Atualização de produto externo)
__global__ void dense_backward_update_weights_kernel(const double* upstream_gradient, const double* input_data,
                                                     double* weights, int num_neurons, int input_size,
                                                     double learning_rate)
{
    // Usar grid 2D é mais intuitivo aqui
    int k = blockIdx.x * blockDim.x + threadIdx.x; // Índice da coluna (input_size)
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Índice da linha (num_neurons)

    if (j < num_neurons && k < input_size) {
        int weight_index = j * input_size + k;
        double gradient_weight = upstream_gradient[j] * input_data[k];
        weights[weight_index] -= learning_rate * gradient_weight;
    }
}


// --- Funções Host que Lançam os Kernels ---
// NOTA: Continuam esperando ponteiros da GPU (d_*)

void dense_forward_math(const double *d_input, const double *d_weights, const double *d_bias,
                        double *d_output, int num_neurons, int input_size)
{
    // Configuração do Kernel
    int threads_per_block = 256;
    int blocks_per_grid = (num_neurons + threads_per_block - 1) / threads_per_block;

    // Lançar Kernel
    dense_forward_kernel<<<blocks_per_grid, threads_per_block>>>(
        d_input, d_weights, d_bias, d_output, num_neurons, input_size);

    // Checar por erros de lançamento e execução (assíncrono!)
    CUDA_CHECK(cudaGetLastError());
    // Para depuração ou se a próxima operação depender desta, sincronize:
    // CUDA_CHECK(cudaDeviceSynchronize());
}

void dense_backward_calc_downstream(const double *d_upstream_gradient, const double *d_weights,
                                    double *d_downstream_gradient, int num_neurons, int input_size)
{
    // Configuração do Kernel
    int threads_per_block = 256;
    int blocks_per_grid = (input_size + threads_per_block - 1) / threads_per_block;

    // Lançar Kernel
    dense_backward_calc_downstream_kernel<<<blocks_per_grid, threads_per_block>>>(
        d_upstream_gradient, d_weights, d_downstream_gradient, num_neurons, input_size);

    CUDA_CHECK(cudaGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize());
}

void dense_backward_update_params(const double *d_upstream_gradient, const double *d_input_data,
                                  double *d_weights, double *d_biases,
                                  int num_neurons, int input_size, double learning_rate)
{
    // --- Atualização dos Biases ---
    int threads_bias = 256;
    int blocks_bias = (num_neurons + threads_bias - 1) / threads_bias;
    dense_backward_update_biases_kernel<<<blocks_bias, threads_bias>>>(
        d_upstream_gradient, d_biases, num_neurons, learning_rate);
    CUDA_CHECK(cudaGetLastError());


    // --- Atualização dos Pesos (Grid 2D) ---
    dim3 threads_weights(16, 16); // Bloco 2D (16*16 = 256 threads)
    dim3 blocks_weights( (input_size + threads_weights.x - 1) / threads_weights.x,
                         (num_neurons + threads_weights.y - 1) / threads_weights.y );

    dense_backward_update_weights_kernel<<<blocks_weights, threads_weights>>>(
        d_upstream_gradient, d_input_data, d_weights, num_neurons, input_size, learning_rate);
    CUDA_CHECK(cudaGetLastError());

    // Sincronizar após as atualizações pode ser importante se o próximo passo
    // no loop de treino depende da conclusão destas escritas.
    // CUDA_CHECK(cudaDeviceSynchronize());
}