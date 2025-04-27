#!/bin/bash

# === Configurações ===
declare -a HIDDEN_SIZES=(32) # Tamanhos da camada oculta a testar
declare -a LEARNING_RATES=(0.01) # Taxas de aprendizado a testar
declare -a EPOCHS=(5 6)          # Número de épocas a testar
SEED=42                             # Semente fixa para todos os runs
OUTPUT_FILE="benchmark_results.csv" # Arquivo para salvar os resultados
GPU_ARCH="sm_86"                    # Substitua pela sua arquitetura (ex: sm_89, sm_75)

# === Compilação ===
echo "Compilando versão CPU..."
gcc main.c neural_network.c dense_layer.c activation_layer.c hw_number.c nn_math.c -o mnist_cpu_bench -lm -lrt -Wall -Wextra -O2
if [ $? -ne 0 ]; then echo "Falha na compilação CPU!"; exit 1; fi

echo "Compilando versão GPU..."
nvcc main.c neural_network.c dense_layer.c activation_layer.c hw_number.c nn_math_cuda.cu -o mnist_gpu_bench -DUSE_CUDA -arch=${GPU_ARCH} -lm -lrt -Xcompiler "-Wall -Wextra -O2"
if [ $? -ne 0 ]; then echo "Falha na compilação GPU!"; exit 1; fi

# === Execução dos Testes ===
echo "Iniciando Benchmarks..."
# Cria cabeçalho do CSV
echo "Backend,HiddenSize,LearningRate,Epochs,Seed,TrainingTime(s),Accuracy(%)" > $OUTPUT_FILE

# Loop pelas configurações
for hs in "${HIDDEN_SIZES[@]}"; do
  for lr in "${LEARNING_RATES[@]}"; do
    for ep in "${EPOCHS[@]}"; do
      echo "--- Executando: HS=$hs LR=$lr Epochs=$ep ---"

      # --- Run CPU ---
      echo "  Rodando CPU..."
      # Executa e captura a saída
      output_cpu=$(./mnist_cpu_bench $hs $lr $ep $SEED)
      # Extrai tempo e acurácia usando grep (alternativa: awk)
      time_cpu=$(echo "$output_cpu" | grep 'TRAINING_TIME:' | cut -d' ' -f2)
      acc_cpu=$(echo "$output_cpu" | grep 'FINAL_ACCURACY:' | cut -d' ' -f2)
      echo "    CPU -> Tempo: ${time_cpu}s, Acurácia: ${acc_cpu}%"
      # Salva no CSV
      echo "CPU,$hs,$lr,$ep,$SEED,$time_cpu,$acc_cpu" >> $OUTPUT_FILE

      # --- Run GPU ---
      echo "  Rodando GPU..."
      # Executa e captura a saída
      output_gpu=$(./mnist_gpu_bench $hs $lr $ep $SEED)
      # Extrai tempo e acurácia
      time_gpu=$(echo "$output_gpu" | grep 'TRAINING_TIME:' | cut -d' ' -f2)
      acc_gpu=$(echo "$output_gpu" | grep 'FINAL_ACCURACY:' | cut -d' ' -f2)
      echo "    GPU -> Tempo: ${time_gpu}s, Acurácia: ${acc_gpu}%"
      # Salva no CSV
      echo "GPU,$hs,$lr,$ep,$SEED,$time_gpu,$acc_gpu" >> $OUTPUT_FILE

    done
  done
done

echo "Benchmarks concluídos. Resultados salvos em $OUTPUT_FILE"