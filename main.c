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
    // --- Parâmetros Fixos ---
    const int    HIDDEN_SIZE = 128;
    const double LEARNING_RATE = 0.2;
    const int    NUM_EPOCHS = 3;
    unsigned int seed = 42; // Semente fixa padrão

    // --- Parâmetro Variável (Número de Camadas Densas) ---
    int num_dense_layers = 2; // Valor padrão (1 oculta + 1 saída)

    // --- Processar Argumentos ---
    // Exemplo: ./programa <num_dense_layers> <seed>
    if (argc >= 2) {
        num_dense_layers = atoi(argv[1]);
        if (num_dense_layers < 2) {
             fprintf(stderr, "Erro: O numero total de camadas densas deve ser pelo menos 2.\n");
             return 1;
        }
    }
     if (argc >= 3) {
         seed = (unsigned int)atoi(argv[2]);
         printf("Usando semente para srand: %u\n", seed);
    } else {
         printf("Usando semente padrao para srand: %u\n", seed);
    }

    // Usar semente
    srand(seed);

    printf("Configuracao: NumDenseLayers=%d, HiddenSize=%d, LR=%.4f, Epochs=%d, Seed=%u\n",
           num_dense_layers, HIDDEN_SIZE, LEARNING_RATE, NUM_EPOCHS, seed);

    // --- 1. Definir Arquitetura e Criar Rede ---
    printf("Criando a rede neural flexivel...\n");
    NeuralNetwork* net = create_network(LEARNING_RATE); // Usa LR fixa
    if (!net) { fprintf(stderr,"Falha ao criar rede\n"); return 1; }

    int input_size = HW_NUM_SIZE; // 784
    int output_size = 10;         // Dígitos 0-9
    int current_input_size = input_size;
    int num_hidden_layers = num_dense_layers - 1; // Número de camadas densas *ocultas*

    printf("Arquitetura: %d", input_size);

    // --- Adicionar Camadas Dinamicamente ---
    Layer *dense_layer = NULL;
    Layer *activation_layer = NULL;

    // Adiciona as camadas ocultas (se houver)
    for (int i = 0; i < num_hidden_layers; ++i) {
        printf(" -> Dense(%d) -> Sigmoid", HIDDEN_SIZE);
        dense_layer = create_dense_layer(current_input_size, HIDDEN_SIZE);
        activation_layer = create_sigmoid_activation_layer(HIDDEN_SIZE);
        if (!dense_layer || !activation_layer ||
            add_layer(net, dense_layer) != 0 ||
            add_layer(net, activation_layer) != 0)
        {
            fprintf(stderr, "Falha ao adicionar camada oculta %d\n", i + 1);
            free_network(net); // Libera o que foi alocado até agora
            return 1;
        }
        current_input_size = HIDDEN_SIZE; // Saída desta camada é entrada da próxima
    }

    // Adiciona a camada de saída final
    printf(" -> Dense(%d) -> Sigmoid\n", output_size);
    dense_layer = create_dense_layer(current_input_size, output_size);
    activation_layer = create_sigmoid_activation_layer(output_size);
     if (!dense_layer || !activation_layer ||
        add_layer(net, dense_layer) != 0 ||
        add_layer(net, activation_layer) != 0)
    {
        fprintf(stderr, "Falha ao adicionar camada de saida\n");
        free_network(net);
        return 1;
    }

    printf("Rede criada com sucesso (%d camadas Layer_t, %d camadas densas).\n", net->num_layers, num_dense_layers);


    // --- 2. Treinar a Rede ---
    int num_training_images = 55000;
    printf("Iniciando treino (%d imagens, %d epocas, lr=%.4f)...\n",
           num_training_images, NUM_EPOCHS, net->learning_rate);

    // Chama a função de treino (que mede o tempo internamente)
    hw_train_flexible_network(net, num_training_images, NUM_EPOCHS); // Usa Epochs fixo


    // --- 3. Testar a Rede ---
    int num_testing_images = 10000;
    printf("\nIniciando teste final (%d imagens)...\n", num_testing_images);
    // Chama a função de teste (que imprime tempo e acurácia)
    hw_test_flexible_network(net, num_testing_images);


    // --- 4. Salvar a Rede Treinada --- (Opcional para benchmark)
    // const char* save_filename = "mnist_network.dat";
    // ... (código de salvar) ...


    // --- 5. Liberar Memória ---
    printf("Liberando a rede neural...\n");
    free_network(net);
    printf("Programa concluido.\n");

    return 0;
}