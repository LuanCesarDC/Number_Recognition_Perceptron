#include <stdio.h>
#include <stdlib.h>
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
