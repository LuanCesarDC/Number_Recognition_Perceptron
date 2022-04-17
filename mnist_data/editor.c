#include <stdio.h>

#define NUM_SIZE 28*28
#define PATH_IMAGES "./mnist_data/train-images.idx3-ubyte"
#define PATH_LABELS "./mnist_data/train-labels.idx1-ubyte"

// handwritten number
typedef struct {
	unsigned char buffer[NUM_SIZE];
	int num;
} hw_number;

void print_ascii(int x) {
	char grayscale[10] = " .:-=+*#%@";
	if(x <= 25)
		printf("%c", grayscale[0]);
	else if(x <= 50)
		printf("%c", grayscale[1]);
	else if(x <= 75)
		printf("%c", grayscale[2]);
	else if(x <= 100)
		printf("%c", grayscale[3]);
	else if(x <= 125)
		printf("%c", grayscale[4]);
	else if(x <= 150)
		printf("%c", grayscale[5]);
	else if(x <= 175)
		printf("%c", grayscale[6]);
	else if(x <= 200)
		printf("%c", grayscale[7]);
	else if(x <= 225)
		printf("%c", grayscale[8]);
	else if(x <= 300)
		printf("%c", grayscale[9]);
}

void 

int main() {
	unsigned char buffer[28];
	FILE * ptr;
	
	ptr = fopen("train-images.idx3-ubyte", "rb");
	fseek(ptr, 16, SEEK_SET);
	
	int x = 0;
	while (fread(buffer, sizeof(buffer), 1, ptr) != 0 && x!=28*100) {
    	for(int i=0;i<28;i++) {
    		printf("%02X ", buffer[i]);
    		//print_ascii(buffer[i]);
    	}
    	printf("\n");
    	
    	x++;
    	if(x%28 == 0)
    		printf("\n\n\n");
	}

	return 0;
}
