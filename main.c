#include "neural_network.h"   // Rede flexível
#include "dense_layer.h"        // Camadas densas
#include "activation_layer.h"   // Camadas de ativação
#include "hw_number.h"        // Funções de treino/teste adaptadas e dados MNIST

#include <stdio.h>
#include <stdlib.h> // Para atoi, atof
#include <time.h>
#include <string.h> // Para strcmp (opcional, para parsing mais robusto)

// Dentro da função main:
int main(int argc, char *argv[]) {
    // --- Valores Padrão ---
    int hidden_size = 64;
    double learning_rate = 0.01;
    int num_epochs = 15;
    unsigned int seed = 0; // Semente fixa para reprodutibilidade

    // --- Processar Argumentos (Exemplo Básico) ---
    // Exemplo: ./programa <hidden_size> <learning_rate> <epochs> <seed>
    if (argc >= 2) {
        hidden_size = atoi(argv[1]);
    }
    if (argc >= 3) {
        learning_rate = atof(argv[2]);
    }
    if (argc >= 4) {
        num_epochs = atoi(argv[3]);
    }
    if (argc >= 5) {
         seed = (unsigned int)atoi(argv[4]);
         printf("Usando semente para srand: %u\n", seed);
    } else {
         printf("Usando semente padrao para srand: %u\n", seed);
    }


    // Usar semente fixa para reprodutibilidade entre runs CPU/GPU
    srand(seed); // <<< MODIFICADO

    printf("Configuracao: Hidden=%d, LR=%.4f, Epochs=%d, Seed=%u\n",
           hidden_size, learning_rate, num_epochs, seed);


    // --- 1. Definir Arquitetura e Criar Rede ---
    printf("Criando a rede neural flexivel...\n");
    // Use a learning_rate lida dos argumentos
    NeuralNetwork* net = create_network(learning_rate);
    if (!net) { /* ... erro ... */ return 1; }

    int input_size = HW_NUM_SIZE; // 784
    int output_size = 10;         // Dígitos 0-9

    printf("Arquitetura: %d -> Dense(%d) -> Sigmoid -> Dense(%d) -> Sigmoid\n",
           input_size, hidden_size, output_size);

    // Adicionar camadas (use hidden_size lido dos argumentos)
    Layer* dense1 = create_dense_layer(input_size, hidden_size);
    Layer* act1 = create_sigmoid_activation_layer(hidden_size);
    Layer* dense2 = create_dense_layer(hidden_size, output_size);
    Layer* act2 = create_sigmoid_activation_layer(output_size);

    if (add_layer(net, dense1) != 0 || add_layer(net, act1) != 0 ||
        add_layer(net, dense2) != 0 || add_layer(net, act2) != 0) {
        /* ... erro ... */ return 1;
    }
    printf("Rede criada com sucesso (%d camadas).\n", net->num_layers);


    // --- 2. Treinar a Rede ---
    int num_training_images = 55000; // Pode ser parametrizado também se desejar
    printf("Iniciando treino (%d imagens, %d epocas, lr=%.4f)...\n",
           num_training_images, num_epochs, net->learning_rate); // Use net->learning_rate

    // Chama a função de treino (que agora mede o tempo internamente)
    hw_train_flexible_network(net, num_training_images, num_epochs);


    // --- 3. Testar a Rede ---
    int num_testing_images = 10000;
    printf("\nIniciando teste final (%d imagens)...\n", num_testing_images);
    // Chama a função de teste (que agora imprime a acurácia no formato parseável)
    hw_test_flexible_network(net, num_testing_images);


    // --- 4. Salvar a Rede Treinada --- (Opcional para benchmark)
    // const char* save_filename = "mnist_network.dat";
    // printf("\nSalvando a rede treinada em %s...\n", save_filename);
    // if (network_save(net, save_filename) != 0) { /* ... erro ... */ }


    // --- 5. Liberar Memória ---
    printf("Liberando a rede neural...\n");
    free_network(net);
    printf("Programa concluido.\n");

    return 0;
}