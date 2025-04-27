
# Rede Neural Flexível em C para MNIST (CPU e CUDA)

## Visão Geral

Este projeto implementa um framework básico para redes neurais artificiais em C puro, com foco em flexibilidade e na demonstração de execução tanto em CPU quanto em GPU (usando CUDA com kernels manuais). Como exemplo, ele treina e testa uma rede neural simples para classificar dígitos manuscritos do dataset MNIST.

O código permite a criação de redes com diferentes arquiteturas (sequências de camadas) e possui backends de computação matemática separados para CPU e GPU.

## Funcionalidades

* **Estrutura Flexível:** Baseada em uma interface genérica de `Layer`, permitindo adicionar diferentes tipos de camadas (Densas, Ativação Sigmoid implementadas).
* **Backend CPU:** Implementação das operações matemáticas (multiplicação de matrizes, etc.) usando C padrão.
* **Backend GPU (CUDA):** Implementação alternativa das operações matemáticas usando kernels CUDA escritos manualmente (sem dependência da biblioteca cuBLAS). A compilação para GPU é opcional.
* **Exemplo MNIST:** Inclui código para carregar, pré-processar, treinar e testar a rede no dataset MNIST.
* **Salvar/Carregar Rede:** Funcionalidade para salvar os pesos treinados da rede em um arquivo. (Funcionalidade de carregar não implementada no `main.c` de exemplo).

## Estrutura dos Arquivos

* `neural_network.h`/`.c`: Define a estrutura `NeuralNetwork` e a interface genérica `Layer`, gerenciando a sequência de camadas e as passagens forward/backward.
* `dense_layer.h`/`.c`: Implementação de uma camada totalmente conectada (densa), incluindo gerenciamento de pesos/biases e lógica para usar backend CPU ou GPU.
* `activation_layer.h`/`.c`: Implementação de uma camada de ativação (Sigmoid neste caso).
* `nn_math.h`: Header que declara as funções matemáticas necessárias para a camada densa.
* `nn_math.c`: Implementação **CPU** das funções matemáticas declaradas em `nn_math.h`.
* `nn_math_cuda.cu`: Implementação **GPU (CUDA)** das funções matemáticas declaradas em `nn_math.h`, usando kernels manuais.
* `hw_number.h`/`.c`: Funções para manipulação do dataset MNIST (carregamento, impressão) e as rotinas de treino/teste adaptadas para a rede flexível.
* `main.c`: Ponto de entrada do programa. Define a arquitetura da rede, adiciona as camadas, e coordena o treino, teste e salvamento.

## Pré-requisitos

* **Compilador C:** GCC ou Clang.
* **Dataset MNIST:** Os arquivos de dados (`train-images.idx3-ubyte`, `train-labels.idx1-ubyte`, `t10k-images.idx3-ubyte`, `t10k-labels.idx1-ubyte`) devem estar em um subdiretório chamado `mnist_data` no mesmo local onde o executável será rodado. Você pode baixá-los de [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/).
* **(Opcional - Para GPU):**
    * Placa de vídeo NVIDIA com suporte a CUDA.
    * NVIDIA CUDA Toolkit instalado (contém o compilador `nvcc` e as bibliotecas CUDA).

## Como Compilar

Compile todos os arquivos fonte juntos. Não é necessário gerar arquivos objeto (`.o`) separadamente.

### Compilação para CPU

Use `gcc`. Este comando utiliza o backend matemático `nn_math.c`.

```bash
gcc main.c neural_network.c dense_layer.c activation_layer.c hw_number.c nn_math.c -o mnist_cpu_direct -lm -Wall -Wextra -O2
```

### Compilação para GPU (CUDA com Kernels Manuais)

Use `nvcc`. Este comando utiliza o backend `nn_math_cuda.cu` e ativa as seções de código específicas para CUDA nos outros arquivos.

**Importante:** Substitua `sm_XX` pela capacidade de computação (Compute Capability) da sua GPU (ex: `sm_75` para arquitetura Turing, `sm_86` para Ampere). Consulte a documentação da NVIDIA ou use o comando `nvidia-smi` para encontrar o valor correto.

```bash
nvcc main.c neural_network.c dense_layer.c activation_layer.c hw_number.c nn_math_cuda.cu -o mnist_gpu_manual_direct -DUSE_CUDA -arch=sm_XX -lm -Xcompiler "-Wall -Wextra -O2"
```

* `-DUSE_CUDA`: Define a macro que ativa o código condicional para CUDA.
* `-arch=sm_XX`: Especifica a arquitetura da GPU alvo. **Substitua `XX`!**
* `-Xcompiler "..."`: Passa flags (`-Wall`, `-Wextra`, `-O2`) para o compilador C host que o `nvcc` utiliza.
* **Não** é necessário usar `-lcublas`, pois estamos usando kernels manuais.

## Como Executar

1.  Certifique-se de que o dataset MNIST está no diretório `mnist_data/`.
2.  Execute o binário compilado correspondente no seu terminal:
    * Para CPU: `./mnist_cpu_direct`
    * Para GPU: `./mnist_gpu_manual_direct`

O programa irá criar a rede, treiná-la por algumas épocas (imprimindo o progresso e a precisão parcial), testá-la no conjunto de teste (imprimindo a precisão final e alguns exemplos) e, por fim, salvar a rede treinada no arquivo `mnist_network.dat`.

## Como Funciona (Resumo)

1.  **Criação:** O `main.c` define a arquitetura criando e adicionando instâncias de `Layer` (Dense, Sigmoid) à `NeuralNetwork`.
2.  **Forward Pass:** A função `network_forward` itera pelas camadas, chamando a função `forward` de cada uma. A saída de uma camada é a entrada da próxima. Para camadas densas na versão GPU, `dense_forward` copia os dados para a GPU, chama a função `dense_forward_math` (que lança o kernel CUDA) e copia o resultado de volta.
3.  **Backward Pass (Backpropagation):** A função `network_backward` calcula o erro inicial na última camada e propaga o gradiente para trás, chamando a função `backward` de cada camada.
    * `activation_backward`: Calcula o gradiente local da função de ativação.
    * `dense_backward`: Calcula o gradiente em relação aos pesos/biases e o gradiente a ser passado para a camada anterior. Atualiza os pesos/biases usando a taxa de aprendizado. Na versão GPU, essas operações ocorrem na GPU através dos kernels manuais, com cópias de dados conforme necessário.
4.  **Treino/Teste:** As funções em `hw_number.c` orquestram o processo de carregar dados, chamar `network_forward`/`network_backward` repetidamente para o treino, e avaliar a precisão no conjunto de teste.