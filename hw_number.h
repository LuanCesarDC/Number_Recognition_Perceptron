#ifndef _HW_NUMBER_H_
#define _HW_NUMBER_H_

#define MODO_ASCII 	0
#define MODO_HEX	1
#define HW_NUM_SIZE 	28*28
#define PATH_IMAGES "./mnist_data/train-images.idx3-ubyte"
#define PATH_LABELS "./mnist_data/train-labels.idx1-ubyte"

// handwritten number
typedef struct {
	unsigned char buffer[HW_NUM_SIZE];
	int digit;
} hw_number;

void print_ascii(int x);

hw_number get_training_image(int index);

void hw_number_print(hw_number image, int modo);

void hw_train_neural_network(char * path, int num, int epochs);

int get_output(neural_network * net);

int * gen_permutation(int num, int * v);

#endif //_HW_NUMBER_H_
