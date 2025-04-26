// ======================================================
#include "nn_math.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h> // Para error checking

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

// Macro para checagem de erro cuBLAS
#define CUBLAS_CHECK(call)                                                                                 \
    do                                                                                                     \
    {                                                                                                      \
        cublasStatus_t status = call;                                                                      \
        if (status != CUBLAS_STATUS_SUCCESS)                                                               \
        {                                                                                                  \
            fprintf(stderr, "cuBLAS Error in %s:%d - %s: Status %d\n", __FILE__, __LINE__, #call, status); \
            exit(EXIT_FAILURE);                                                                            \
        }                                                                                                  \
    } while (0)

// --- Funções GPU ---
// NOTA IMPORTANTE: Estas funções agora recebem ponteiros para memória DA GPU!
// A cópia Host -> Device e Device -> Host deve ser feita ANTES e DEPOIS
// de chamar estas funções (geralmente dentro de dense_forward/dense_backward).

// Handle global para cuBLAS (criar na inicialização do programa)
extern cublasHandle_t cublas_handle; // Precisa ser criado em algum lugar (ex: main)

void dense_forward_math(const double *d_input, const double *d_weights, const double *d_bias,
                        double *d_output, int num_neurons, int input_size)
{
    // Operação: d_output = d_bias + alpha * d_weights * d_input
    // Usando cuBLAS: dgemv (Matrix-Vector multiply)
    // y = alpha*A*x + beta*y
    // Aqui: output = 1.0 * weights * input + 1.0 * bias (inicialmente copiando bias para output)

    const double alpha = 1.0;
    const double beta = 1.0;

    // 1. Copiar bias para o vetor de saída d_output (pois dgemv adiciona a y)
    CUDA_CHECK(cudaMemcpy(d_output, d_bias, num_neurons * sizeof(double), cudaMemcpyDeviceToDevice));

    // 2. Calcular d_weights * d_input e adicionar a d_output
    //    cuBLAS assume column-major por padrão. Se weights é row-major (como no C),
    //    precisamos transpor a operação ou a matriz.
    //    Op: Usar dgemv com transposição: output = bias + W * input
    //    W é (num_neurons x input_size). input é (input_size x 1). output é (num_neurons x 1).
    //    A=weights (M=num_neurons, N=input_size), x=input, y=output
    CUBLAS_CHECK(cublasDgemv(cublas_handle,
                             CUBLAS_OP_N, // Operação da Matriz A (sem transpor, assumindo layout correto ou pré-transposto)
                             num_neurons, // rows of A
                             input_size,  // cols of A
                             &alpha,
                             d_weights,   // A (matriz de pesos na GPU)
                             num_neurons, // Leading dimension of A (lda)
                             d_input,     // x (vetor de entrada na GPU)
                             1,           // Incremento de x
                             &beta,
                             d_output, // y (vetor de saída na GPU)
                             1));      // Incremento de y
}

void dense_backward_calc_downstream(const double *d_upstream_gradient, const double *d_weights,
                                    double *d_downstream_gradient, int num_neurons, int input_size)
{
    // Operação: d_downstream = d_upstream * d_weights
    // Usando cuBLAS: dgemv (transposto)
    // y = alpha*A^T*x + beta*y
    // Aqui: downstream = 1.0 * weights^T * upstream + 0.0 * downstream
    // A = weights (M=num_neurons, N=input_size). x = upstream (M=num_neurons). y = downstream (N=input_size).

    const double alpha = 1.0;
    const double beta = 0.0; // Zera o vetor de saída antes de calcular

    CUBLAS_CHECK(cublasDgemv(cublas_handle,
                             CUBLAS_OP_T, // Transpõe a matriz A (weights)
                             num_neurons, // rows of A original
                             input_size,  // cols of A original
                             &alpha,
                             d_weights,           // A
                             num_neurons,         // lda
                             d_upstream_gradient, // x
                             1,
                             &beta,
                             d_downstream_gradient, // y
                             1));
}

void dense_backward_update_params(const double *d_upstream_gradient, const double *d_input_data,
                                  double *d_weights, double *d_biases,
                                  int num_neurons, int input_size, double learning_rate)
{
    // Operação 1: grad_weights = upstream_gradient^T * input_data (produto externo - dger)
    // Operação 2: grad_bias = upstream_gradient
    // Operação 3: weights -= learning_rate * grad_weights (daxpy ou kernel customizado)
    // Operação 4: biases -= learning_rate * grad_bias (daxpy)

    const double alpha = -learning_rate; // Para subtração nas atualizações

    // --- Atualização dos Biases (Operação 2 e 4 combinadas) ---
    // biases = biases - learning_rate * upstream_gradient
    // Usando daxpy: y = alpha*x + y
    // Aqui: biases = (-learning_rate) * upstream_gradient + biases
    CUBLAS_CHECK(cublasDaxpy(cublas_handle,
                             num_neurons,         // Número de elementos
                             &alpha,              // -learning_rate
                             d_upstream_gradient, // x
                             1,
                             d_biases, // y (será modificado)
                             1));

    // --- Atualização dos Pesos (Operação 1 e 3) ---
    // weights = weights - learning_rate * (upstream_gradient^T * input_data)
    // Usando dger: A = alpha*x*y^T + A
    // Aqui: weights = (-learning_rate) * upstream^T * input + weights
    // x = upstream (M=num_neurons), y = input (N=input_size), A = weights (M x N)
    CUBLAS_CHECK(cublasDger(cublas_handle,
                            num_neurons,         // M (rows of A)
                            input_size,          // N (cols of A)
                            &alpha,              // -learning_rate
                            d_upstream_gradient, // x
                            1,
                            d_input_data, // y
                            1,
                            d_weights,     // A (será modificado)
                            num_neurons)); // lda
}