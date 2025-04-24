#include "neural_network.h"
#include "activation_layer.h"
#include "dense_layer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h> // Para memcpy

#define INITIAL_NETWORK_CAPACITY 4

// --- Implementação das Funções da Rede ---

NeuralNetwork *create_network(double learning_rate)
{
	NeuralNetwork *net = (NeuralNetwork *)malloc(sizeof(NeuralNetwork));
	if (!net)
		return NULL;

	net->layers = (Layer **)malloc(INITIAL_NETWORK_CAPACITY * sizeof(Layer *));
	if (!net->layers)
	{
		free(net);
		return NULL;
	}
	net->num_layers = 0;
	net->capacity = INITIAL_NETWORK_CAPACITY;
	net->learning_rate = learning_rate;
	return net;
}

void free_network(NeuralNetwork *net)
{
	if (!net)
		return;
	// Libera cada camada usando seu próprio ponteiro free_layer
	for (int i = 0; i < net->num_layers; ++i)
	{
		if (net->layers[i] && net->layers[i]->free_layer)
		{
			net->layers[i]->free_layer(net->layers[i]);
		}
	}
	free(net->layers); // Libera o array de ponteiros
	free(net);		   // Libera a struct da rede
}

// Função auxiliar para aumentar a capacidade do array de camadas
int ensure_network_capacity(NeuralNetwork *net)
{
	if (net->num_layers >= net->capacity)
	{
		int new_capacity = net->capacity * 2;
		Layer **new_layers_ptr = (Layer **)realloc(net->layers, new_capacity * sizeof(Layer *));
		if (!new_layers_ptr)
		{
			fprintf(stderr, "Erro ao realocar memoria para ponteiros de camadas\n");
			return -1;
		}
		net->layers = new_layers_ptr;
		net->capacity = new_capacity;
	}
	return 0;
}

// Adiciona uma camada (já criada) à rede
int add_layer(NeuralNetwork *net, Layer *layer)
{
	if (!net || !layer)
		return -1;

	// Verifica se o tamanho da entrada da nova camada bate com a saída da anterior
	if (net->num_layers > 0)
	{
		Layer *prev_layer = net->layers[net->num_layers - 1];
		if (prev_layer->output_size != layer->input_size)
		{
			fprintf(stderr, "Erro: Incompatibilidade de tamanho entre camadas! Saida anterior=%d, Entrada nova=%d\n",
					prev_layer->output_size, layer->input_size);
			return -1; // Importante checar isso!
		}
	}

	if (ensure_network_capacity(net) != 0)
		return -1; // Garante espaço

	net->layers[net->num_layers] = layer;
	net->num_layers++;
	return 0;
}

// Executa o feedforward através de todas as camadas
int network_forward(NeuralNetwork *net, const double *input)
{
	if (!net || !input)
		return -1;
	if (net->num_layers == 0)
		return 0; // Rede vazia

	double *current_output = NULL;		 // Buffer para a saída da camada atual
	const double *current_input = input; // Entrada inicial é a da rede

	for (int i = 0; i < net->num_layers; ++i)
	{
		Layer *current_layer = net->layers[i];

		// O buffer de saída da camada atual é o buffer 'activations' dela
		current_output = current_layer->activations;

		// Chama o forward da camada atual
		if (current_layer->forward(current_layer, current_input, current_output) != 0)
		{
			fprintf(stderr, "Erro durante o forward na camada %d\n", i);
			return -1;
		}

		// A saída desta camada (current_output) se torna a entrada da próxima
		current_input = current_output;
	}
	return 0; // Sucesso
}

// Executa o backpropagation através de todas as camadas
int network_backward(NeuralNetwork *net, const double *expected_output)
{
	if (!net || !expected_output)
		return -1;
	int n_layers = net->num_layers;
	if (n_layers == 0)
		return 0;

	// --- Passo 1: Calcular o erro inicial na última camada ---
	Layer *last_layer = net->layers[n_layers - 1];
	// O gradiente inicial (upstream) para a última camada.
	// Depende da função de perda e da ativação da última camada.
	// Exemplo: Para perda MSE e última ativação Sigmoid:
	// gradient = (activation - expected) * sigmoid_deriv(activation)
	double *initial_gradient = (double *)malloc(last_layer->output_size * sizeof(double));
	if (!initial_gradient)
		return -1;

	for (int i = 0; i < last_layer->output_size; ++i)
	{
		double activation = last_layer->activations[i];
		// Assumindo que a última camada foi Sigmoid e a perda é MSE
		// (Idealmente, a função de perda seria separada)
		double error_signal = activation - expected_output[i];
		// Assumindo que a última camada *é* uma Sigmoid (ou tem ativação sigmoid embutida)
		// Se a ultima camada for só Densa, a derivada da ativação (que viria depois) é que entra aqui.
		// VAMOS ASSUMIR QUE A ÚLTIMA CAMADA É DE ATIVAÇÃO SIGMOID por simplicidade:
		initial_gradient[i] = error_signal * sigmoid_deriv_from_output(activation); // Gradiente delta inicial
		// Copia o gradiente inicial para o buffer de gradiente downstream da última camada,
		// pois ele será o upstream da penúltima.
		last_layer->downstream_gradient[i] = initial_gradient[i];
	}

	// Ponteiro para o gradiente que VEM da camada seguinte (começa com o inicial)
	const double *upstream_gradient = initial_gradient; // Para a penúltima camada

	// --- Passo 2: Propagar o erro para trás ---
	for (int i = n_layers - 2; i >= 0; --i)
	{ // Itera da penúltima para a primeira
		Layer *current_layer = net->layers[i];
		Layer *next_layer = net->layers[i + 1]; // Camada de onde veio o gradiente

		// Gradiente que VEM da camada seguinte
		upstream_gradient = next_layer->downstream_gradient;

		// Buffer onde esta camada escreverá o gradiente para a camada anterior
		double *downstream_gradient_buffer = current_layer->downstream_gradient;

		// Entrada que esta camada recebeu durante o forward
		const double *input_data_from_forward = current_layer->input_data_buffer;

		// Chama o backward da camada atual
		if (current_layer->backward(current_layer, upstream_gradient,
									input_data_from_forward,
									downstream_gradient_buffer, net->learning_rate) != 0)
		{
			fprintf(stderr, "Erro durante o backward na camada %d\n", i);
			free(initial_gradient);
			return -1;
		}
		// O downstream_gradient calculado por esta camada se torna o upstream para a próxima iteração (anterior)
		// (Não precisa fazer nada, pois já foi escrito no buffer correto e será lido na próxima iteração)
	}

	free(initial_gradient); // Libera o buffer do gradiente inicial
	return 0;				// Sucesso
}

int network_save(NeuralNetwork *net, const char *filename)
{
	if (!net || !filename)
		return -1;

	FILE *fp = fopen(filename, "wb"); // Abrir em modo binário para escrita
	if (!fp)
	{
		perror("Erro ao abrir arquivo para salvar rede");
		return -1;
	}

	// 1. Escrever o número de camadas
	if (fwrite(&net->num_layers, sizeof(int), 1, fp) != 1)
	{
		perror("Erro ao escrever numero de camadas");
		fclose(fp);
		return -1;
	}

	// 2. Iterar e salvar cada camada
	for (int i = 0; i < net->num_layers; ++i)
	{
		Layer *layer = net->layers[i];

		// 2.1 Determinar e Escrever o Tipo da Camada
		//     Precisamos de uma forma de saber o tipo. Vamos usar um truque:
		//     Verificar qual função 'save' está atribuída. Não é ideal, um enum LayerType
		//     na struct Layer seria melhor, mas vamos adaptar ao código atual.
		LayerType type_to_save;
		if (layer->save == dense_save)
		{ // Compara ponteiros de função
			type_to_save = LAYER_DENSE;
		}
		else if (layer->save == activation_save)
		{											 // Assumindo que só temos Sigmoid por enquanto
													 // Precisaríamos diferenciar Sigmoid/ReLU se tivéssemos ambos
			type_to_save = LAYER_ACTIVATION_SIGMOID; // Assumir Sigmoid
		}
		else
		{
			fprintf(stderr, "Erro: Tipo de camada desconhecido para salvar na camada %d\n", i);
			fclose(fp);
			return -1; // Tipo desconhecido
		}
		if (fwrite(&type_to_save, sizeof(LayerType), 1, fp) != 1)
		{ // Escreve o enum/int do tipo
			perror("Erro ao escrever tipo da camada");
			fclose(fp);
			return -1;
		}

		// 2.2 Escrever tamanhos input/output
		if (fwrite(&layer->input_size, sizeof(int), 1, fp) != 1)
		{ /* erro */
			fclose(fp);
			return -1;
		}
		if (fwrite(&layer->output_size, sizeof(int), 1, fp) != 1)
		{ /* erro */
			fclose(fp);
			return -1;
		}

		// 2.3 Chamar a função 'save' específica da camada
		if (layer->save(layer, fp) != 0)
		{
			fprintf(stderr, "Erro ao salvar dados especificos da camada %d (tipo %d)\n", i, type_to_save);
			fclose(fp);
			return -1;
		}
		printf("Camada %d (Tipo %d, %d->%d) salva.\n", i, type_to_save, layer->input_size, layer->output_size);

	} // Fim for camadas

	printf("Rede salva com sucesso em %s\n", filename);
	fclose(fp);
	return 0; // Sucesso
}

// Retorna ponteiro para as ativações da última camada
double *get_network_output(NeuralNetwork *net)
{
	if (!net || net->num_layers == 0)
		return NULL;
	return net->layers[net->num_layers - 1]->activations;
}
