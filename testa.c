#include "neural_network.h"
#include "hw_number.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
	srand(time(NULL));
	
	//double v[2] = {0.521, 0.215};
	float v[2] = {0.5, 0.5};
	//hw_number x = get_training_image(1);
	neural_network * n = create_neural_network(1, 1, 1, 3);
	array_to_input(n, v);
	feedforward(n);
	
	printf("h -> w: %lf  bias: %lf\no -> w: %lf  bias: %lf\n", n->hidden->neurons->weights[0],
												n->hidden->neurons->bias,
												n->output.neurons->weights[0],
												n->output.neurons->bias);
	
	printa_camadas(n);
	printf("\n\nCusto: %lf", layer_cost(&n->output, 2));
	
}
