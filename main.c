#include "neural_network.h"   // Rede flexível
#include "dense_layer.h"        // Camadas densas
#include "activation_layer.h"   // Camadas de ativação
#include "hw_number.h"        // Funções de treino/teste adaptadas e dados MNIST

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
    srand(time(NULL));

    // --- 1. Definir Arquitetura e Criar Rede ---
    printf("Criando a rede neural flexivel...\n");
    double learning_rate = 0.01; // Taxa de aprendizado inicial
    NeuralNetwork* net = create_network(learning_rate);
    if (!net) {
        fprintf(stderr, "Falha ao criar rede\n");
        return 1;
    }

    int input_size = HW_NUM_SIZE; // 784
    // Aumentar tamanho da camada oculta para mais capacidade
    int hidden_size = 64;         // <<< AUMENTADO DE 16 PARA 64
    int output_size = 10;         // Dígitos 0-9

    printf("Arquitetura: %d -> Dense(%d) -> Sigmoid -> Dense(%d) -> Sigmoid\n",
           input_size, hidden_size, output_size);

    // Adicionar camadas
    Layer* dense1 = create_dense_layer(input_size, hidden_size);
    Layer* act1 = create_sigmoid_activation_layer(hidden_size);
    Layer* dense2 = create_dense_layer(hidden_size, output_size);
    Layer* act2 = create_sigmoid_activation_layer(output_size);

    if (add_layer(net, dense1) != 0 || add_layer(net, act1) != 0 ||
        add_layer(net, dense2) != 0 || add_layer(net, act2) != 0) {
        fprintf(stderr, "Falha ao adicionar camadas na rede\n");
        free_network(net);
        return 1;
    }
    printf("Rede criada com sucesso (%d camadas).\n", net->num_layers);


    // --- 2. Treinar a Rede ---
    int num_training_images = 55000; // <<< USAR MAIS IMAGENS (Máx 60000)
    int num_epochs = 15;             // <<< MAIS ÉPOCAS
    printf("Iniciando treino robusto (%d imagens, %d epocas, lr=%.3f)...\n",
           num_training_images, num_epochs, learning_rate);

    hw_train_flexible_network(net, num_training_images, num_epochs);


    // --- 3. Testar a Rede ---
    int num_testing_images = 10000; // <<< TESTAR NO CONJUNTO COMPLETO
    printf("\nIniciando teste final (%d imagens)...\n", num_testing_images);
    hw_test_flexible_network(net, num_testing_images);


    // --- 4. Salvar a Rede Treinada ---  <<< ADICIONADO
    const char* save_filename = "mnist_network.dat";
    printf("\nSalvando a rede treinada em %s...\n", save_filename);
    if (network_save(net, save_filename) != 0) {
        fprintf(stderr, "Erro ao salvar a rede!\n");
    }


    // --- 5. Liberar Memória ---
    printf("Liberando a rede neural...\n");
    free_network(net);
    printf("Programa concluido.\n");

    return 0;
}