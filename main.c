#include "hw_number.h"
#include "neural_network.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
	unsigned char buffer[28];
	
	for(int i=0;i<50000;i++) {
		hw_number a = get_training_image(i);
		hw_number_print(a, MODO_ASCII);
		printf("\n%d\n", i);
	}
	

	return 0;
}
