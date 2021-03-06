#include "neural_network.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define LEARNING_RATE 0.0025

float sigmoid(float x) {
	return 1/(1+exp(-x));
}

float sigmoid_deriv(float x) {
	return x*(1-x);  // OBS isso pensando que x já é sigmoid(k)
					 // caso contrario seria return sigmoid(x)*(1-sigmoid(x))
}

void create_neuron(neuron * node, int connections) {
	int i;
	node->weights = (double *) malloc(connections*sizeof(double));
	
	for(i=0;i<connections;i++) {
		node->weights[i] = ((float)rand() / (float)(RAND_MAX) * 10.0) - 5;
	}
	node->bias = ((float)rand() / (float)(RAND_MAX) * 10.0) - 5;
}

neural_network * create_neural_network(int input_size, int num_hidden, int hidden_size, int output_size) {
	int i, j;
	neural_network * net = (neural_network *) malloc(sizeof(neural_network));

	// Camada de entrada	
	net->input.size = input_size;
	net->input.neurons = (neuron *) malloc(input_size*sizeof(neuron));

	// Camada interna
	net->hidden = (layer *) malloc(num_hidden*sizeof(layer));
	net->num_hidden = num_hidden;
	for(i=0;i<num_hidden;i++) {
		net->hidden[i].size = hidden_size;
		net->hidden[i].neurons = (neuron *) malloc(hidden_size*sizeof(neuron));
		
		int num_connections = (i == 0) ? (input_size):(hidden_size);
		for(j=0;j<hidden_size;j++) {
			create_neuron(&net->hidden[i].neurons[j], num_connections);
		}
	}
	
	// Camada externa
	net->output.size = output_size;
	net->output.neurons = (neuron *) malloc(output_size*sizeof(neuron));
	for(i=0;i<output_size;i++) {
		create_neuron(&net->output.neurons[i], hidden_size);
	}
	return net;
}

void free_neural_network(neural_network * net) {
	int i, j, k;
	// Libera os neuronios da primeira camada
	free(net->input.neurons);
	
	// Libera as camadas internas
	for(i=0;i<net->num_hidden;i++) {
		for(j=0;j<net->hidden[i].size;j++) {
			free(net->hidden[i].neurons[j].weights);
		}
		free(net->hidden[i].neurons);
	}
	free(net->hidden);
	
	// Libera a ultima camada
	for(i=0;i<net->output.size;i++) {
		free(net->output.neurons[i].weights);
	}
	free(net->output.neurons);
}

void array_to_input(neural_network * net, unsigned char * array) {
	int i;
	for(i=0;i<net->input.size;i++) {
		net->input.neurons[i].activ = (float) array[i]/255.0;
		//net->input.neurons[i].activ = array[i];
	}
}

void feedforward(neural_network * net) {
	int i, j, k;
	
	// First hidden layer
	for(i=0;i<net->hidden[0].size;i++) {
		double soma = 0;
		for(j=0;j<net->input.size;j++) {
			soma += net->hidden[0].neurons[i].weights[j] * net->input.neurons[j].activ;
		}
		net->hidden[0].neurons[i].activ = sigmoid(soma + net->hidden[0].neurons[i].bias);
	}
	
	// Hidden Layers
	for(i=1;i<net->num_hidden;i++) { // Itera nas camadas internas a partir da segunda
		for(j=0;j<net->hidden[i].size;j++) { // Itera nos neuronios de hidden[i]
			double soma = 0;
			for(k=0;k<net->hidden[i-1].size;k++) { // Itera nos pesos de cada neurônio de hidden[i]
				soma += net->hidden[i].neurons[j].weights[k] * net->hidden[i-1].neurons[k].activ;
			}
			net->hidden[i].neurons[j].activ = sigmoid(soma + net->hidden[i].neurons[j].bias);
		}
	}
	
	// Camada Externa
	for(i=0;i<net->output.size;i++) {
		double soma = 0;
		for(j=0;j<net->hidden[net->num_hidden-1].size;j++) {
			soma += net->output.neurons[i].weights[j] * net->hidden[net->num_hidden-1].neurons[j].activ;
		}
		net->output.neurons[i].activ = sigmoid(soma + net->output.neurons[i].bias);
	}
	
}

// batch_size = 1 primeiramente
// Dps tem q implementar o batch_size
void backpropagation(neural_network * net, double * expected) {
	int i, j, k, iter;
	

	// Erros camada saída
	for(i=0;i<net->output.size;i++) 
		net->output.neurons[i].error = net->output.neurons[i].activ - expected[i];
	
	// Erros ultima camada interna
	for(i=0;i<net->hidden[net->num_hidden-1].size;i++) {
		double soma = 0;
		for(j=0;j<net->output.size;j++)
			soma += net->output.neurons[j].error * net->output.neurons[j].weights[i];
		net->hidden[net->num_hidden-1].neurons[i].error = soma;
	}
	
	// Erros restantes camadas internas
	for(k=net->num_hidden-2;k>=0;k--) { // Itera sobre as camadas exceto a ultima
		if(net->num_hidden < 2) break;
		
		for(i=0;i<net->hidden[k].size;i++) { // Itera sobre os neuronios da camada "k"
			double soma = 0;
			for(j=0;j<net->hidden[k+1].size;j++)  // Itera sobre os neuronios da camada "k+1"
				soma += net->hidden[k+1].neurons[j].error * net->hidden[k+1].neurons[j].weights[i];
			
			net->hidden[k].neurons[i].error = soma;
		}
	}
	
	
	
	// Atualização pesos da primeira camada interna
	for(i=0;i<net->hidden[0].size;i++) { // Itera sobre os neuronios de hidden[0]
		
		double activ = net->hidden[0].neurons[i].activ;				// a(L)
		double error = net->hidden[0].neurons[i].error; 				//(a(L) - y)
		for(j=0;j<net->input.size;j++) {
			double activ_ant = net->input.neurons[j].activ;  		// a(L-1)
			
			net->hidden[0].neurons[i].weights[j] += -1 * LEARNING_RATE * 
											  		activ_ant *
											  		error * 
											  		sigmoid_deriv(activ);
		}
		// Atualiza o Bias
		net->hidden[0].neurons[i].bias += -1 *  LEARNING_RATE *
												error *
												sigmoid_deriv(activ);	
	}
	
	// Atualização pesos das demais camadas internas
	for(k=1;k<net->num_hidden;k++) {
		if(net->num_hidden < 2) break;
			
		for(i=0;i<net->hidden[k].size;i++) { // Itera sobre os neuronios de hidden[k]
			double activ = net->hidden[k].neurons[i].activ; 			// a(L)			
			double error = net->hidden[k].neurons[i].error; 				//(a(L) - y)
			for(j=0;j<net->hidden[k-1].size;j++) { // Itera sobre os pesos de hidden[k]
				double activ_ant = net->hidden[k-1].neurons[j].activ;  		// a(L-1)
				
				net->hidden[k].neurons[i].weights[j] += -1 * LEARNING_RATE * 
												  		activ_ant *
												  		error * 
												  		sigmoid_deriv(activ);
				
			}
		// Atualiza o Bias
		net->hidden[k].neurons[i].bias += -1 *  LEARNING_RATE *
												error *
												sigmoid_deriv(activ);	
		}
	} 

	// Atualização da camada de saida
	for(i=0;i<net->output.size;i++) { // Itera sobre os neuronios de output
		double activ = net->output.neurons[i].activ; 				// a(L)
		double error = net->output.neurons[i].error; 			//(a(L) - y)
		for(j=0;j<net->hidden[net->num_hidden-1].size;j++) {
			double activ_ant = net->hidden[net->num_hidden-1].neurons[j].activ;  		// a(L-1)
		//	printf("inc: %lf\n", -1 * LEARNING_RATE * activ_ant * error * sigmoid_deriv(activ));
			
			
			net->output.neurons[i].weights[j] += -1 * LEARNING_RATE * 
											  	 activ_ant *
											  	 error * 
											  	 sigmoid_deriv(activ);
			
		}
		// Atualiza o Bias
		net->output.neurons[i].bias += -1 *  LEARNING_RATE *
												error *
												sigmoid_deriv(activ);	
	}
}

void save_neural_network(neural_network * net, char * path) {
	FILE * ptr;
	size_t data_written;
	int i, j, k;
	
	ptr = fopen(path, "wb");
	if(ptr == NULL) {
		fprintf(stderr, "Não foi possível abrir o arquivo para escrita.\n");
    	exit(1);
	}
	// Guarda o tamanho de cada camada e o numero de camadas internas
	fwrite(&net->input.size, sizeof(int), 1, ptr);
	fwrite(&net->num_hidden, sizeof(int), 1, ptr);
	fwrite(&net->hidden[0].size, sizeof(int), 1, ptr);
	fwrite(&net->output.size, sizeof(int), 1, ptr);
	
	// Guarda os pesos de cada camada interna
	for(i=0;i<net->hidden[0].size;i++) {
		fwrite(&net->hidden[0].neurons[i].bias, sizeof(double), 1, ptr);
		for(j=0;j<net->input.size;j++) {
			fwrite(&net->hidden[0].neurons[i].weights[j], sizeof(double), 1, ptr);
		}
	}
	for(i=1;i<net->num_hidden;i++) {
		for(j=0;j<net->hidden[i].size;j++) {
			fwrite(&net->hidden[i].neurons[j].bias, sizeof(double), 1, ptr);
			for(k=0;k<net->hidden[i-1].size;k++) {
				fwrite(&net->hidden[i].neurons[j].weights[k], sizeof(double), 1, ptr);
			}
		}
	}
	
	// Guarda os pesos da camada de saída
	for(i=0;i<net->output.size;i++) {
		fwrite(&net->output.neurons[i].bias, sizeof(double), 1, ptr);
		for(j=0;j<net->hidden[net->num_hidden-1].size;j++) {
			fwrite(&net->output.neurons[i].weights[j], sizeof(double), 1, ptr);
		}
	}
	
	fclose(ptr);
	
}

neural_network * load_neural_network(char * path) {
	FILE * ptr;
	neural_network * net;
	size_t data_read;
	int i, j, k;
	int input_size, num_hidden, hidden_size, output_size;
	
	ptr = fopen(path, "rb");
	if(ptr == NULL) {
		fprintf(stderr, "Não foi possível abrir o arquivo para escrita.\n");
    	exit(1);
	}
	
	// Carrega o tamanho de cada camada e o numero de camadas internas
	fread(&input_size, sizeof(int), 1, ptr);
	fread(&num_hidden, sizeof(int), 1, ptr);
	fread(&hidden_size, sizeof(int), 1, ptr);
	fread(&output_size, sizeof(int), 1, ptr);
	
	net = create_neural_network(input_size, num_hidden, hidden_size, output_size);
	
	// Carrega os pesos de cada camada interna
	for(i=0;i<net->hidden[0].size;i++) {
		fread(&net->hidden[0].neurons[i].bias, sizeof(double), 1, ptr);
		for(j=0;j<net->input.size;j++) {
			fread(&net->hidden[0].neurons[i].weights[j], sizeof(double), 1, ptr);
		}
	}
	for(i=1;i<net->num_hidden;i++) {
		for(j=0;j<net->hidden[i].size;j++) {
			fread(&net->hidden[i].neurons[j].bias, sizeof(double), 1, ptr);
			for(k=0;k<net->hidden[i-1].size;k++) {
				fread(&net->hidden[i].neurons[j].weights[k], sizeof(double), 1, ptr);
			}
		}
	}
	
	// Carrega os pesos da camada de saída
	for(i=0;i<net->output.size;i++) {
		fread(&net->output.neurons[i].bias, sizeof(double), 1, ptr);
		for(j=0;j<net->hidden[net->num_hidden-1].size;j++) {
			fread(&net->output.neurons[i].weights[j], sizeof(double), 1, ptr);
		}
	}
	
	fclose(ptr);

	return net;
}

void printa_camadas(neural_network * net) {
	int i, j;
	
	
	for(i=0;i<net->input.size;i++) {
		printf("I%d: %.5lf ", i, net->input.neurons[i].activ);
		if(i%10 == 0 && i > 0)
			printf("\n");
	}
	printf("\n");
	for(i=0;i<net->num_hidden;i++) {
		for(j=0;j<net->hidden[i].size;j++) {
			printf("H%d%d: %.5lf ", i, j, net->hidden[i].neurons[j].activ);
			if(j%8 == 0 && j > 0)
				printf("\n");
		}
		printf("\n");
	}
	for(i=0;i<net->output.size;i++) {
		printf("O%d: %.3lf ", i, net->output.neurons[i].activ);
	}
}

