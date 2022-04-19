#include "neural_network.h"
#include "hw_number.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
	srand(time(NULL));
	double acertos = 0, total = 0;
	neural_network * net  = create_neural_network(28*28, 2, 16, 10);
	//neural_network * net  = load_neural_network("teste92.dat");

	/*for(int i=0;i<100;i++) {
		
		hw_number data = get_training_image(i);
		array_to_input(net, data.buffer);
		feedforward(net);
		
		hw_number_print(data, MODO_ASCII);
		printf("REDE_NEURAL: O digito acima é %d\n", get_output(net));
		printf("RESPOSTA: %d\n", data.digit);
		
				
				
	}*/
	
	hw_train_neural_network("teste.dat", 59999, 800);
	free_neural_network(net);
}
