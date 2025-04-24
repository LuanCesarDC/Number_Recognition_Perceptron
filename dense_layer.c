#include "dense_layer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h> // Para memcpy

// Função auxiliar para inicializar pesos (exemplo)
void initialize_weights(double* weights, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        // Inicialização pequena e aleatória (melhores métodos existem, como Xavier/He)
        weights[i] = ((double)rand() / RAND_MAX) * 0.2 - 0.1;
    }
}

// --- Implementação das Funções da Interface ---

int dense_forward(Layer* layer, const double* input, double* output_buffer) {
    if (!layer || !layer->internal_data || !input || !output_buffer) return -1;

    // 1. Recuperar dados específicos da camada Densa
    DenseLayerData* data = (DenseLayerData*)layer->internal_data;

    // 2. Salvar a entrada recebida (necessária para o backward)
    //    A alocação do buffer input_data_buffer deve ser feita na criação da Layer
    memcpy(layer->input_data_buffer, input, layer->input_size * sizeof(double));

    // 3. Calcular a soma ponderada: output = input * weights^T + bias
    //    (output_buffer aqui guarda a soma ponderada, ANTES da ativação)
    for (int j = 0; j < data->num_neurons; ++j) { // Para cada neurônio desta camada
        output_buffer[j] = data->biases[j];     // Começa com o bias
        for (int k = 0; k < data->input_size; ++k) { // Para cada entrada
            // Acessa o peso: weights[neuronio_destino * tamanho_entrada + indice_entrada]
            output_buffer[j] += input[k] * data->weights[j * data->input_size + k];
        }
    }

    // 4. A *saída final* da camada (ativação) será armazenada no buffer 'activations' da Layer genérica.
    //    Isso será preenchido pela *próxima* camada (se for uma camada de ativação) ou aqui mesmo se a ativação for integrada.
    //    Neste modelo, a camada Densa SÓ calcula a soma ponderada. A ativação é separada.
    //    Copiamos a soma ponderada para o buffer de ativações temporariamente.
    memcpy(layer->activations, output_buffer, layer->output_size * sizeof(double));


    return 0; // Sucesso
}

int dense_backward(Layer* layer, const double* upstream_gradient,
                   const double* input_data_from_forward, // Este é layer->input_data_buffer salvo no forward
                   double* downstream_gradient_buffer, double learning_rate) {
    if (!layer || !layer->internal_data || !upstream_gradient || !input_data_from_forward || !downstream_gradient_buffer) return -1;

    DenseLayerData* data = (DenseLayerData*)layer->internal_data;

    // Gradiente que chega (upstream_gradient) é d(Loss)/d(Output_Desta_Camada)
    // Assumindo que a camada seguinte (que enviou o gradiente) JÁ multiplicou pela derivada da sua ativação.
    // Então, upstream_gradient aqui é o delta de erro para a *soma ponderada* desta camada.

    // 1. Calcular o gradiente para passar para a camada anterior (downstream_gradient)
    //    gradient_anterior_k = SUM_j (upstream_gradient_j * weight_jk)
    //    onde k é neurônio da camada anterior, j é neurônio desta camada.
    for (int k = 0; k < data->input_size; ++k) { // Para cada neurônio da camada anterior
        downstream_gradient_buffer[k] = 0.0;
        for (int j = 0; j < data->num_neurons; ++j) { // Para cada neurônio desta camada
            downstream_gradient_buffer[k] += upstream_gradient[j] * data->weights[j * data->input_size + k];
        }
    }
    // Salvar este gradiente calculado no buffer da Layer genérica
     memcpy(layer->downstream_gradient, downstream_gradient_buffer, layer->input_size * sizeof(double));


    // 2. Calcular o gradiente dos pesos e biases e atualizá-los
    //    gradient_weight_jk = upstream_gradient_j * input_k
    //    gradient_bias_j = upstream_gradient_j
    for (int j = 0; j < data->num_neurons; ++j) { // Para cada neurônio desta camada
        for (int k = 0; k < data->input_size; ++k) { // Para cada entrada/peso
            double weight_gradient = upstream_gradient[j] * input_data_from_forward[k];
            // Atualizar peso
            data->weights[j * data->input_size + k] -= learning_rate * weight_gradient;
        }
        // Atualizar bias
        data->biases[j] -= learning_rate * upstream_gradient[j];
    }

    return 0; // Sucesso
}

int dense_save(Layer* layer, FILE* fp) {
    if (!layer || !layer->internal_data || !fp) return -1;
    DenseLayerData* data = (DenseLayerData*)layer->internal_data;

    // Ordem: biases, depois weights
    // Escreve biases (num_neurons doubles)
    if (fwrite(data->biases, sizeof(double), data->num_neurons, fp) != data->num_neurons) {
        perror("Erro ao escrever biases da camada densa");
        return -1;
    }
    // Escreve weights (num_neurons * input_size doubles)
    size_t num_weights = (size_t)data->num_neurons * data->input_size;
    if (fwrite(data->weights, sizeof(double), num_weights, fp) != num_weights) {
         perror("Erro ao escrever pesos da camada densa");
         return -1;
    }
    return 0; // Sucesso
}

void free_dense_layer(Layer* layer) {
    if (!layer) return;
    if (layer->internal_data) {
        DenseLayerData* data = (DenseLayerData*)layer->internal_data;
        free(data->weights);
        free(data->biases);
        free(data); // Libera a struct de dados específicos
    }
    // Libera buffers da Layer genérica
    free(layer->activations);
    free(layer->input_data_buffer);
    free(layer->downstream_gradient);
    // Libera a própria struct Layer genérica
    free(layer);
}

// --- Função Fábrica ---
Layer* create_dense_layer(int input_size, int num_neurons) {
    // 1. Alocar memória para dados específicos
    DenseLayerData* data = (DenseLayerData*)malloc(sizeof(DenseLayerData));
    if (!data) return NULL;
    data->input_size = input_size;
    data->num_neurons = num_neurons;
    data->weights = (double*)malloc(num_neurons * input_size * sizeof(double));
    data->biases = (double*)malloc(num_neurons * sizeof(double));

    // 2. Alocar memória para a Layer genérica
    Layer* layer = (Layer*)malloc(sizeof(Layer));
    if (!layer || !data->weights || !data->biases) {
        free(data->weights); free(data->biases); free(data);
        free(layer); // Pode ser NULL se a alocação falhou
        return NULL;
    }

    // 3. Alocar buffers na Layer genérica
    layer->activations = (double*)malloc(num_neurons * sizeof(double));
    layer->input_data_buffer = (double*)malloc(input_size * sizeof(double));
    layer->downstream_gradient = (double*)malloc(input_size * sizeof(double)); // Gradiente que sai

    if (!layer->activations || !layer->input_data_buffer || !layer->downstream_gradient) {
         free(data->weights); free(data->biases); free(data);
         free(layer->activations); free(layer->input_data_buffer); free(layer->downstream_gradient);
         free(layer);
         return NULL;
    }


    // 4. Inicializar pesos e biases
    initialize_weights(data->weights, num_neurons, input_size);
    for (int i = 0; i < num_neurons; ++i) data->biases[i] = 0.0; // Inicializa biases com 0

    // 5. Configurar a Layer genérica
    layer->internal_data = data; // Aponta para os dados específicos
    layer->input_size = input_size;
    layer->output_size = num_neurons;
    // Configurar os ponteiros de função!
    layer->forward = dense_forward;
    layer->backward = dense_backward;
    layer->free_layer = free_dense_layer;
    layer->save = dense_save;

    return layer; // Retorna o ponteiro para a Layer genérica configurada
}