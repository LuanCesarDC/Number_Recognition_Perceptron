#ifndef FLEXIBLE_NEURAL_NETWORK_H_
#define FLEXIBLE_NEURAL_NETWORK_H_

#include <stdlib.h>
#include <stdio.h>

// --- DEFINIÇÃO DO ENUM LayerType --- <<< ADICIONAR ESTA SEÇÃO
typedef enum
{
    LAYER_UNKNOWN = 0, // Um valor padrão/erro
    LAYER_DENSE,
    LAYER_ACTIVATION_SIGMOID
    // Adicione outros tipos conforme necessário (ex: LAYER_ACTIVATION_RELU)
} LayerType;

typedef struct Layer_s
{
    int (*forward)(struct Layer_s *layer, const double *input, double *output_buffer);
    int (*backward)(struct Layer_s *layer, const double *upstream_gradient,
                    const double *input_data_from_forward,
                    double *downstream_gradient_buffer, double learning_rate);
    void (*free_layer)(struct Layer_s *layer);
    int (*save)(struct Layer_s *layer, FILE *fp);

    // Dados da Camada:
    void *internal_data;
    int input_size;
    int output_size;

    // Buffers genéricos:
    double *activations;
    double *input_data_buffer;
    double *downstream_gradient;

    // Opcional, mas recomendado para salvar/carregar de forma robusta:
    // LayerType type; // <<< Seria melhor adicionar este campo

} Layer;

// ... (struct NeuralNetwork e protótipos de função como antes) ...
typedef struct
{
    Layer **layers;
    int num_layers;
    int capacity;
    double learning_rate;
} NeuralNetwork;

NeuralNetwork *create_network(double learning_rate);
void free_network(NeuralNetwork *net);
int add_layer(NeuralNetwork *net, Layer *layer);
int network_forward(NeuralNetwork *net, const double *input);
int network_backward(NeuralNetwork *net, const double *expected_output);
double *get_network_output(NeuralNetwork *net);
int network_save(NeuralNetwork *net, const char *filename);
// NeuralNetwork* network_load(const char* filename);

#endif // FLEXIBLE_NEURAL_NETWORK_H_