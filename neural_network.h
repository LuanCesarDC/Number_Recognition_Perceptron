#ifndef _NEURAL_NETWORK_H_
#define _NEURAL_NETWORK_H_

#define BIAS -1

typedef struct {
	double activ;
	double error;
	double bias;
	double * weights;
} neuron;

typedef struct {
	neuron * neurons;
	int size;
} layer;

typedef struct {
	layer input;
	layer * hidden;
	layer output;
	int num_hidden;
} neural_network;

float sigmoid(float x);

void create_neuron(neuron * node, int connections);

neural_network * create_neural_network(int input_size, int num_hidden, int hidden_size, int output_size);

void free_neural_network(neural_network * net);

void array_to_input(neural_network * net, unsigned char * array);

void feedforward(neural_network * net);

void backpropagation(neural_network * net, double * expected);

void save_neural_network(neural_network * net, char * path);

neural_network * load_neural_network(char * path);

void printa_camadas(neural_network * net);


#endif //_NEURAL_NETWORK_H_
