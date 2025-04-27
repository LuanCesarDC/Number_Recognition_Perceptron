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
        // Itera pela entrada k para calcular o produto escalar
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
    int k = blockIdx.x * blockDim.x + threadIdx.x; // Índice do gradiente de saída (para camada anterior, tamanho input_size)

    if (k < input_size) {
        double sum = 0.0;
        // Itera pelos neurônios j da camada atual
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

    // Verifica limites 2D
    if (j < num_neurons && k < input_size) {
        int weight_index = j * input_size + k;
        double gradient_weight = upstream_gradient[j] * input_data[k]; // Gradiente = delta * entrada
        weights[weight_index] -= learning_rate * gradient_weight; // Atualiza peso
    }
}


// --- Funções Host que Lançam os Kernels ---
// Envolver as definições das funções chamadas por C com extern "C"
extern "C" {

// Forward: Calcula output = (input * weights^T) + bias
// input: (1 x input_size) -> d_input
// weights: (num_neurons x input_size) -> d_weights
// bias: (num_neurons x 1) -> d_bias
// output: (num_neurons x 1) -> d_output (Armazena o resultado)
void dense_forward_math(const double *d_input, const double *d_weights, const double *d_bias,
                        double *d_output, int num_neurons, int input_size)
{
    // Configuração do Kernel 1D
    int threads_per_block = 256;
    // Calcula o número de blocos necessários para cobrir todos os neurônios
    int blocks_per_grid = (num_neurons + threads_per_block - 1) / threads_per_block;

    // Lançar Kernel
    dense_forward_kernel<<<blocks_per_grid, threads_per_block>>>(
        d_input, d_weights, d_bias, d_output, num_neurons, input_size);

    // Checar por erros de lançamento (importante!)
    CUDA_CHECK(cudaGetLastError());
    // Opcional: Sincronizar se a próxima operação depender do resultado estar pronto
    // CUDA_CHECK(cudaDeviceSynchronize());
}

// Backward (Cálculo do gradiente para camada anterior):
// downstream_gradient = upstream_gradient * weights
// upstream_gradient: (1 x num_neurons) -> d_upstream_gradient
// weights: (num_neurons x input_size) -> d_weights
// downstream_gradient: (1 x input_size) -> d_downstream_gradient (Armazena o resultado)
void dense_backward_calc_downstream(const double *d_upstream_gradient, const double *d_weights,
                                    double *d_downstream_gradient, int num_neurons, int input_size)
{
    // Configuração do Kernel 1D (lança threads para cada elemento do gradiente de saída)
    int threads_per_block = 256;
    int blocks_per_grid = (input_size + threads_per_block - 1) / threads_per_block;

    // Lançar Kernel
    dense_backward_calc_downstream_kernel<<<blocks_per_grid, threads_per_block>>>(
        d_upstream_gradient, d_weights, d_downstream_gradient, num_neurons, input_size);

    // Checar por erros de lançamento
    CUDA_CHECK(cudaGetLastError());
    // Opcional: Sincronizar
    // CUDA_CHECK(cudaDeviceSynchronize());
}

// Backward (Atualização de pesos e biases):
// grad_weights = upstream_gradient^T * input_data
// grad_bias = upstream_gradient
// weights -= learning_rate * grad_weights
// biases -= learning_rate * grad_bias
// upstream_gradient: (1 x num_neurons) -> d_upstream_gradient
// input_data: (1 x input_size) -> d_input_data (entrada que gerou a ativação no forward)
// weights: (num_neurons x input_size) -> d_weights (Será modificado)
// biases: (num_neurons x 1) -> d_biases (Será modificado)
void dense_backward_update_params(const double *d_upstream_gradient, const double *d_input_data,
                                  double *d_weights, double *d_biases,
                                  int num_neurons, int input_size, double learning_rate)
{
    // --- Atualização dos Biases ---
    int threads_bias = 256;
    int blocks_bias = (num_neurons + threads_bias - 1) / threads_bias;
    dense_backward_update_biases_kernel<<<blocks_bias, threads_bias>>>(
        d_upstream_gradient, d_biases, num_neurons, learning_rate);
    // Checa erro APÓS o lançamento do kernel de bias
    CUDA_CHECK(cudaGetLastError());


    // --- Atualização dos Pesos (Grid 2D) ---
    // Define o tamanho do bloco 2D (ex: 16x16 = 256 threads por bloco)
    dim3 threads_weights(16, 16);
    // Calcula o número de blocos em cada dimensão
    dim3 blocks_weights( (input_size + threads_weights.x - 1) / threads_weights.x,   // Blocos na dimensão X (cols)
                         (num_neurons + threads_weights.y - 1) / threads_weights.y ); // Blocos na dimensão Y (rows)

    dense_backward_update_weights_kernel<<<blocks_weights, threads_weights>>>(
        d_upstream_gradient, d_input_data, d_weights, num_neurons, input_size, learning_rate);
    // Checa erro APÓS o lançamento do kernel de pesos
    CUDA_CHECK(cudaGetLastError());

    // Pode ser útil sincronizar aqui se a próxima iteração do loop de treino
    // depende que essas atualizações estejam completas.
    // CUDA_CHECK(cudaDeviceSynchronize());
}

} // Fim do extern "C"