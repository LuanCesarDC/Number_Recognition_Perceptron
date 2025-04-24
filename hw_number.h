#ifndef _HW_NUMBER_H_
#define _HW_NUMBER_H_

// Inclui a definição da nova estrutura NeuralNetwork
#include "neural_network.h" // <<< ADICIONAR ESTA LINHA

#define MODO_ASCII      0
#define MODO_HEX        1
#define HW_NUM_SIZE     (28*28) // Melhor usar parênteses
#define PATH_IMAGES     "./mnist_data/train-images.idx3-ubyte"
#define PATH_LABELS     "./mnist_data/train-labels.idx1-ubyte"
#define PATH_T_IMAGES   "./mnist_data/t10k-images.idx3-ubyte"
#define PATH_T_LABELS   "./mnist_data/t10k-labels.idx1-ubyte"

// handwritten number (struct permanece a mesma)
typedef struct {
    unsigned char buffer[HW_NUM_SIZE];
    int digit;
} hw_number;

void print_ascii(int x);
hw_number get_training_image(int index);
hw_number get_testing_image(int index);
void hw_number_print(hw_number image, int modo);

// Funções de treino/teste agora operam na nova rede.
// O parâmetro 'path' para salvar/carregar não é mais usado diretamente aqui
// (a menos que implementemos save/load para a nova estrutura).
// Retornaremos void para simplificar, pois não estamos mais retornando taxa de erro antiga.
void hw_train_flexible_network(NeuralNetwork* net, int num_images, int epochs); // Passa a rede criada externamente
void hw_test_flexible_network(NeuralNetwork* net, int num_images);  // Passa a rede criada externamente

// Nova função auxiliar para obter a previsão (substitui get_output)
int get_prediction_from_output(NeuralNetwork * net);

// Função de permutação permanece a mesma
int * gen_permutation(int num, int * v);

#endif //_HW_NUMBER_H_