#ifndef _NEURAL_NETWORK_H_
#define _NEURAL_NETWORK_H_

#define BIAS -1

typedef struct {
	double activ;
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
	int num_hidden;
	layer output;
} neural_network;

float sigmoid(float x);

void create_neuron(neuron * node, int connections);

neural_network * create_neural_network(int input_size, int num_hidden, int hidden_size, int output_size);

void array_to_input(neural_network * net, float * array);

void feedforward(neural_network * net);


void printa_camadas(neural_network * net);

double layer_cost(layer * lay, int required);

#endif //_NEURAL_NETWORK_H_
