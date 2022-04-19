#include "neural_network.h"
#include "hw_number.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
	srand(time(NULL));
	double acertos = 0, total = 0;
	//neural_network * net  = create_neural_network(28*28, 2, 16, 10);
	neural_network * net  = load_neural_network("teste.dat");

	for(int i=0;i<59999;i++) {
		
		hw_number data = get_training_image(i);
		array_to_input(net, data.buffer);
		feedforward(net);
		
		if(get_output(net) == data.digit)
			acertos++;
		total++;
	}
	
	printf("Acertos: %d\nTotal: %d\nPercent.: %.3lf\n", (int) acertos, (int) total, 100.0*acertos/total);
	
	//hw_train_neural_network("teste.dat", 59999);
	
}
