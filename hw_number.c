#include <stdio.h>
#include <stdlib.h>
#include <string.h> // Para memcpy se necessário
#include <time.h>   // Para srand nos testes

// Headers da Nova Estrutura
#include "neural_network.h"   // Rede flexível e Layer genérica
#include "dense_layer.h"      // Para criar camadas densas
#include "activation_layer.h" // Para criar camadas de ativação (Sigmoid)

// Header original do hw_number
#include "hw_number.h"

// Funções get_training_image, get_testing_image, print_ascii, hw_number_print
// permanecem EXATAMENTE as mesmas do seu arquivo original.
// ... (copie essas funções do seu hw_number.c original aqui) ...

void print_ascii(int x)
{
    char grayscale[10] = " .:-=+*#%@";
    if (x <= 25)
        printf("%c", grayscale[0]);
    else if (x <= 50)
        printf("%c", grayscale[1]);
    else if (x <= 75)
        printf("%c", grayscale[2]);
    else if (x <= 100)
        printf("%c", grayscale[3]);
    else if (x <= 125)
        printf("%c", grayscale[4]);
    else if (x <= 150)
        printf("%c", grayscale[5]);
    else if (x <= 175)
        printf("%c", grayscale[6]);
    else if (x <= 200)
        printf("%c", grayscale[7]);
    else if (x <= 225)
        printf("%c", grayscale[8]);
    else if (x <= 300)
        printf("%c", grayscale[9]);
}

hw_number get_training_image(int index)
{
    int flag1, flag2;
    hw_number num;

    FILE *images = fopen(PATH_IMAGES, "rb");
    FILE *labels = fopen(PATH_LABELS, "rb");
    // Tratamento de erro básico (arquivos não encontrados)
    if (!images || !labels)
    {
        fprintf(stderr, "Erro: Nao foi possivel abrir arquivos MNIST de treino em %s / %s\n", PATH_IMAGES, PATH_LABELS);
        exit(1);
    }

    flag1 = fseek(images, 16 + index * HW_NUM_SIZE, SEEK_SET); // Skip file header and go to index
    flag2 = fseek(labels, 8 + index, SEEK_SET);                // SKip file header and go to label

    if (flag1 || flag2)
    {
        fprintf(stderr, "Erro ao encontrar a imagem de treino (fseek falhou)\n");
        fclose(images);
        fclose(labels);
        exit(1);
    }
    // Remover a verificação >= 60000 pois o chamador deve controlar isso
    // if(index >= 60000) { ... }

    // Ler dados
    if (fread(num.buffer, sizeof(unsigned char), HW_NUM_SIZE, images) != HW_NUM_SIZE)
    {
        fprintf(stderr, "Erro ao ler dados da imagem de treino %d\n", index);
        fclose(images);
        fclose(labels);
        exit(1);
    }
    int label_char = fgetc(labels);
    if (label_char == EOF)
    {
        fprintf(stderr, "Erro ao ler rotulo da imagem de treino %d\n", index);
        fclose(images);
        fclose(labels);
        exit(1);
    }
    num.digit = label_char;

    fclose(images);
    fclose(labels);
    return num;
}

hw_number get_testing_image(int index)
{
    int flag1, flag2;
    hw_number num;

    FILE *images = fopen(PATH_T_IMAGES, "rb");
    FILE *labels = fopen(PATH_T_LABELS, "rb");
    if (!images || !labels)
    {
        fprintf(stderr, "Erro: Nao foi possivel abrir arquivos MNIST de teste em %s / %s\n", PATH_T_IMAGES, PATH_T_LABELS);
        exit(1);
    }

    flag1 = fseek(images, 16 + index * HW_NUM_SIZE, SEEK_SET); // Skip file header and go to index
    flag2 = fseek(labels, 8 + index, SEEK_SET);                // SKip file header and go to label

    if (flag1 || flag2)
    {
        fprintf(stderr, "Erro ao encontrar a imagem de teste (fseek falhou)\n");
        fclose(images);
        fclose(labels);
        exit(1);
    }
    // Remover verificação >= 10000

    if (fread(num.buffer, sizeof(unsigned char), HW_NUM_SIZE, images) != HW_NUM_SIZE)
    {
        fprintf(stderr, "Erro ao ler dados da imagem de teste %d\n", index);
        fclose(images);
        fclose(labels);
        exit(1);
    }
    int label_char = fgetc(labels);
    if (label_char == EOF)
    {
        fprintf(stderr, "Erro ao ler rotulo da imagem de teste %d\n", index);
        fclose(images);
        fclose(labels);
        exit(1);
    }
    num.digit = label_char;

    fclose(images);
    fclose(labels);
    return num;
}

void hw_number_print(hw_number image, int modo)
{
    for (int i = 0; i < 28; i++)
    {
        for (int j = 0; j < 28; j++)
        {
            if (modo == MODO_ASCII)
                print_ascii(image.buffer[i * 28 + j]);
            else if (modo == MODO_HEX)
                printf("%02X", image.buffer[i * 28 + j]);
        }
        printf("\n");
    }
}

// Função gen_permutation permanece a mesma
int *gen_permutation(int num, int *v)
{
    int i;
    for (i = 0; i < num; i++)
    {
        v[i] = i;
    }
    // Embaralhamento Fisher-Yates
    for (i = num - 1; i > 0; i--)
    { // Correção: ir até i > 0
        int x = rand() % (i + 1);
        int aux;

        aux = v[i];
        v[i] = v[x];
        v[x] = aux;
    }
    return v;
}

// --- Nova Função Auxiliar ---
// Encontra o índice (dígito) com maior valor na saída da rede
int get_prediction_from_output(NeuralNetwork *net)
{
    if (!net || net->num_layers == 0)
        return -1; // Ou algum erro

    double *output_vector = get_network_output(net); // Pega as ativações da última camada
    if (!output_vector)
        return -1;

    Layer *last_layer = net->layers[net->num_layers - 1];
    int output_size = last_layer->output_size;

    int max_idx = 0;
    for (int i = 1; i < output_size; ++i)
    {
        if (output_vector[i] > output_vector[max_idx])
        {
            max_idx = i;
        }
    }
    return max_idx;
}

// --- Função de Treinamento Adaptada ---
// Recebe a rede já criada e configurada
void hw_train_flexible_network(NeuralNetwork *net, int num_images_to_train, int epochs)
{
    // --- Verificações iniciais (como antes) ---
    if (!net)
    {
        fprintf(stderr, "Erro: Rede Neural nao inicializada para treino.\n");
        return;
    }
    printf("Iniciando treinamento com %d imagens por %d epocas.\n", num_images_to_train, epochs);
    printf("Taxa de aprendizado: %f\n", net->learning_rate);
    if (net->num_layers == 0)
    {
        fprintf(stderr, "Erro: Rede sem camadas para treinar.\n");
        return;
    }
    int input_size = net->layers[0]->input_size;
    int output_size = net->layers[net->num_layers - 1]->output_size;

    // --- Alocações (como antes) ---
    double *input_vector = (double *)malloc(input_size * sizeof(double));
    double *expected_vector = (double *)malloc(output_size * sizeof(double));
    int *permutation_indices = (int *)malloc(num_images_to_train * sizeof(int));

    if (!input_vector || !expected_vector || !permutation_indices)
    {
        fprintf(stderr, "Erro: Falha ao alocar memoria para treino.\n");
        free(input_vector);
        free(expected_vector);
        free(permutation_indices);
        return;
    }

    // >>> INÍCIO: Código de Medição de Tempo <<<
    struct timespec start_time, end_time; // Estruturas para tempo
    double elapsed_time;

    clock_gettime(CLOCK_MONOTONIC, &start_time); // Marca tempo inicial
    // >>> FIM: Código de Medição de Tempo <<<

    // Loop de épocas
    for (int k = 0; k < epochs; k++)
    {
        int total_epoch = 0;
        int acertos_epoch = 0;

        gen_permutation(num_images_to_train, permutation_indices);
        printf("###### [Epoch %d / %d] ######\n", k + 1, epochs);

        // Loop sobre as imagens de treino (na ordem permutada)
        for (int i = 0; i < num_images_to_train; i++)
        {
            // ... (código interno do loop de imagens como antes) ...
            // 1. Preparar Entrada
            // 2. Preparar Saída Esperada
            // 3. Feedforward
            // 4. Backpropagation
            // 5. Verificar Acerto
            // Imprimir progresso...

        } // Fim loop imagens

        printf("\nEpoch %d Concluida. Acertos: %d / Total: %d / Precisao: %.3f%%\n\n",
               k + 1, acertos_epoch, total_epoch, 100.0 * acertos_epoch / total_epoch);

    } // Fim loop épocas

    // >>> INÍCIO: Cálculo e Impressão do Tempo <<<
    clock_gettime(CLOCK_MONOTONIC, &end_time); // Marca tempo final

    // Calcular tempo decorrido em segundos
    elapsed_time = (end_time.tv_sec - start_time.tv_sec);
    elapsed_time += (end_time.tv_nsec - start_time.tv_nsec) / 1000000000.0;

    // Imprimir tempo de treinamento (formato fácil de parsear)
    printf("TRAINING_TIME: %.4f\n", elapsed_time); // <<< ESSA LINHA É IMPORTANTE
    // >>> FIM: Cálculo e Impressão do Tempo <<<

    // Libera memória alocada para os buffers
    free(input_vector);
    free(expected_vector);
    free(permutation_indices);

    printf("Treinamento concluido.\n");
}

// --- Função de Teste Adaptada ---
// Recebe a rede já criada e (presumivelmente) treinada
void hw_test_flexible_network(NeuralNetwork *net, int num_images_to_test)
{
    if (!net)
    {
        fprintf(stderr, "Erro: Rede Neural nao inicializada para teste.\n");
        return;
    }
    printf("Iniciando teste com %d imagens.\n", num_images_to_test);

    // Determina tamanhos a partir da rede
    if (net->num_layers == 0)
    {
        fprintf(stderr, "Erro: Rede sem camadas para testar.\n");
        return;
    }
    int input_size = net->layers[0]->input_size;
    // int output_size = net->layers[net->num_layers - 1]->output_size; // <<< LINHA REMOVIDA

    // Aloca buffer reutilizável para entrada
    double *input_vector = (double *)malloc(input_size * sizeof(double));
    if (!input_vector)
    {
        fprintf(stderr, "Erro: Falha ao alocar memoria para teste.\n");
        return;
    }

    int total_test = 0;
    int acertos_test = 0;

    // Loop sobre as imagens de teste
    for (int i = 0; i < num_images_to_test; i++)
    {
        hw_number data = get_testing_image(i); // Pega imagem de teste

        // 1. Preparar Entrada: Normalizar pixels
        for (int px = 0; px < input_size; ++px)
        {
            input_vector[px] = (double)data.buffer[px] / 255.0;
        }

        // 2. Feedforward
        if (network_forward(net, input_vector) != 0)
        {
            fprintf(stderr, "Erro no forward durante teste na imagem %d.\n", i);
            continue; // Pula para a próxima imagem
        }

        // 3. Obter Previsão e Verificar Acerto
        int prediction = get_prediction_from_output(net);
        if (prediction == data.digit)
        {
            acertos_test++;
        }
        total_test++;

        // Opcional: Imprimir a imagem e a previsão/resposta para algumas amostras
        if (i < 10)
        { // Imprime para as 10 primeiras
            printf("\n--- Imagem Teste %d ---\n", i);
            hw_number_print(data, MODO_ASCII);
            printf("REDE PREVIU: %d | CORRETO: %d %s\n", prediction, data.digit, (prediction == data.digit ? "(OK)" : "(ERRO)"));
        }
        else if (i == 10)
        {
            printf("... (demais imagens omitidas para brevidade)\n");
        }

    } // Fim loop imagens teste

    // Calcular acurácia
    double final_accuracy = (total_test > 0) ? (100.0 * acertos_test / total_test) : 0.0;

    // Imprime resultado final do teste (como antes)
    printf("\nTeste Concluido.\n");
    printf("Acertos: %d / Total: %d / Precisao Final: %.3f%%\n\n",
           acertos_test, total_test, final_accuracy);

    // Imprimir acurácia final (formato fácil de parsear)
    printf("FINAL_ACCURACY: %.4f\n", final_accuracy); // <<< ADICIONADO

    free(input_vector); // Libera buffer de entrada
}