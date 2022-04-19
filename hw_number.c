#include <stdio.h>
#include <stdlib.h>
#include "neural_network.h"
#include "hw_number.h"


void print_ascii(int x) {
	char grayscale[10] = " .:-=+*#%@";
	if		(x <=  25)	printf("%c", grayscale[0]);
	else if	(x <=  50) 	printf("%c", grayscale[1]);
	else if	(x <=  75) 	printf("%c", grayscale[2]);
	else if	(x <= 100)	printf("%c", grayscale[3]);
	else if	(x <= 125)	printf("%c", grayscale[4]);
	else if	(x <= 150)	printf("%c", grayscale[5]);
	else if	(x <= 175)	printf("%c", grayscale[6]);
	else if	(x <= 200)	printf("%c", grayscale[7]);
	else if	(x <= 225)	printf("%c", grayscale[8]);
	else if	(x <= 300)	printf("%c", grayscale[9]);
}

hw_number get_training_image(int index) {
	int flag1, flag2;
	hw_number num;
	
	FILE * images = fopen(PATH_IMAGES, "rb");
	FILE * labels = fopen(PATH_LABELS, "rb");
	flag1 = fseek(images, 16+index*HW_NUM_SIZE, SEEK_SET);	// Skip file header and go to index
	flag2 = fseek(labels, 8+index, SEEK_SET);  				// SKip file header and go to label
	
	if(flag1 || flag2) {
		printf("Erro ao encontrar a imagem de treino\n");
		exit(1);
	}
	if(index >= 60000) {
		printf("Os indices das imagens de treino vão de 0 a 59999!\n");
		exit(1);
	}
	
	fread(num.buffer, sizeof(num.buffer), 1, images);
	num.digit = fgetc(labels);
	
	
	fclose(images);
	fclose(labels);
	return num;
}

hw_number get_testing_image(int index) {
	int flag1, flag2;
	hw_number num;
	
	FILE * images = fopen(PATH_T_IMAGES, "rb");
	FILE * labels = fopen(PATH_T_LABELS, "rb");
	flag1 = fseek(images, 16+index*HW_NUM_SIZE, SEEK_SET);	// Skip file header and go to index
	flag2 = fseek(labels, 8+index, SEEK_SET);  				// SKip file header and go to label
	
	if(flag1 || flag2) {
		printf("Erro ao encontrar a imagem de teste\n");
		exit(1);
	}
	if(index >= 60000) {
		printf("Os indices das imagens de teste vão de 0 a 59999!\n");
		exit(1);
	}
	
	fread(num.buffer, sizeof(num.buffer), 1, images);
	num.digit = fgetc(labels);
	
	fclose(images);
	fclose(labels);
	return num;
}

void hw_number_print(hw_number image, int modo) {

	for(int i=0;i<28;i++) {
		for(int j=0;j<28;j++) {
			if(modo == MODO_ASCII)
				print_ascii(image.buffer[i*28 + j]);
			else if(modo == MODO_HEX)
				printf("%02X", image.buffer[i*28 + j]);
		}
		printf("\n");
	}
}

double hw_train_neural_network(char * path, int num, int epochs) {
	neural_network * net = load_neural_network(path); //= create_neural_network(28*28, 2, 16, 10);
	int i, k;
	int total, acertos;
	double max = 0.8;
	
	for(k=0;k<epochs;k++) {
		int v[num];
		total = 0;
		acertos = 0;
		gen_permutation(num, v);
		printf("###### [Epoch %d] ######\n", k);
		for(i=0;i<num;i++) {
			double expected[10] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};	
			hw_number data = get_training_image(v[i]);
			expected[data.digit] = 1.0;
			
			array_to_input(net, data.buffer);
			feedforward(net);
			backpropagation(net, expected);
			
			//printf("[Iteração %d]\nEsperado: %d\nSaída: %d\n", v[i], data.digit, get_output(net));
			if(data.digit == get_output(net))
				acertos++;
			total++;
		}
		if((double) acertos/total > max) {
			save_neural_network(net, path);
			max = (double) acertos/total;	
		}
		printf("Acertos: %d\nTotal: %d\nPercent.: %.3lf\n\n", acertos, total, 100.0*acertos/total);
	}
	
	
	free_neural_network(net);
	return (double) total/acertos;
}

void hw_test_neural_network(char * path, int num) {
	int i;
	int total = 0, acertos = 0;
	neural_network * net = load_neural_network(path);
	
	for(i=0;i<num;i++) {
			double expected[10] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};	
			hw_number data = get_training_image(i);
			expected[data.digit] = 1.0;
			
			array_to_input(net, data.buffer);
			feedforward(net);
			
			if(data.digit == get_output(net))
				acertos++;
			hw_number_print(data, MODO_ASCII);
			printf("REDE_NEURAL: Dígito -> %d\nRESPOSTA: Dígito -> %d\n\n", get_output(net), data.digit);	
			total++;
	}
	printf("Acertos: %d\nTotal: %d\nPercent.: %.3lf%%\n\n", acertos, total, 100.0*acertos/total);	
}

int get_output(neural_network * net) {
	int i, imax = 0;
	double max = -1.0;
	for(i=0;i<10;i++) {
		if(net->output.neurons[i].activ > max) {
			max = net->output.neurons[i].activ;
			imax = i;
		}
	}
	return imax;
}

int * gen_permutation(int num, int * v) {
	int i;
	for(i=0;i<num;i++) {
		v[i] = i;
	}
	for(i=num-1;i>=0;i--) {
		int x = rand() % (i+1);
		int aux;
		
		aux = v[i];
		v[i] = v[x];
		v[x] = aux;
	}
	return v;
}

