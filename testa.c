#include "neural_network.h"
#include "hw_number.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
	srand(time(NULL));
	
	//double v[2] = {0.521, 0.215};
	
	hw_number x = get_training_image(1);
	neural_network * n = create_neural_network(784, 2, 16, 10);
	array_to_input(n, x.buffer);
	feedforward(n);
	
	printa_camadas(n);
	
}
