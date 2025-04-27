#!/bin/bash

# === Configurações ===
declare -a HIDDEN_SIZES=(32)
declare -a LEARNING_RATES=(0.1)
declare -a EPOCHS=(5 6)
SEED=42
OUTPUT_FILE="benchmark_results_full.csv" # Novo nome de arquivo
GPU_ARCH="sm_86" # Substitua pela sua arquitetura

# === Compilação (com -lrt) ===
echo "Compilando versão CPU..."
gcc main.c neural_network.c dense_layer.c activation_layer.c hw_number.c nn_math.c -o mnist_cpu_bench -lm -lrt -Wall -Wextra -O2
if [ $? -ne 0 ]; then echo "Falha na compilação CPU!"; exit 1; fi

echo "Compilando versão GPU..."
nvcc main.c neural_network.c dense_layer.c activation_layer.c hw_number.c nn_math_cuda.cu -o mnist_gpu_bench -DUSE_CUDA -arch=${GPU_ARCH} -lm -lrt -Xcompiler "-Wall -Wextra -O2"
if [ $? -ne 0 ]; then echo "Falha na compilação GPU!"; exit 1; fi

# === Execução dos Testes ===
echo "Iniciando Benchmarks..."
# Cria cabeçalho do CSV (adiciona InferenceTime)
echo "Backend,HiddenSize,LearningRate,Epochs,Seed,TrainingTime(s),InferenceTime(s),Accuracy(%)" > $OUTPUT_FILE

# Loop pelas configurações
for hs in "${HIDDEN_SIZES[@]}"; do
  for lr in "${LEARNING_RATES[@]}"; do
    for ep in "${EPOCHS[@]}"; do
      echo "--- Executando: HS=$hs LR=$lr Epochs=$ep ---"

      # --- Run CPU ---
      echo "  Rodando CPU..."
      output_cpu=$(./mnist_cpu_bench $hs $lr $ep $SEED)
      # Extrai tempo de treino, tempo de inferência e acurácia
      time_cpu=$(echo "$output_cpu" | grep 'TRAINING_TIME:' | cut -d' ' -f2)
      inf_time_cpu=$(echo "$output_cpu" | grep 'INFERENCE_TIME:' | cut -d' ' -f2) # <<< NOVO
      acc_cpu=$(echo "$output_cpu" | grep 'FINAL_ACCURACY:' | cut -d' ' -f2)
      echo "    CPU -> Treino: ${time_cpu}s, Inferência: ${inf_time_cpu}s, Acurácia: ${acc_cpu}%"
      # Salva no CSV (adiciona inf_time_cpu)
      echo "CPU,$hs,$lr,$ep,$SEED,$time_cpu,$inf_time_cpu,$acc_cpu" >> $OUTPUT_FILE # <<< MODIFICADO

      # --- Run GPU ---
      echo "  Rodando GPU..."
      output_gpu=$(./mnist_gpu_bench $hs $lr $ep $SEED)
      # Extrai tempo de treino, tempo de inferência e acurácia
      time_gpu=$(echo "$output_gpu" | grep 'TRAINING_TIME:' | cut -d' ' -f2)
      inf_time_gpu=$(echo "$output_gpu" | grep 'INFERENCE_TIME:' | cut -d' ' -f2) # <<< NOVO
      acc_gpu=$(echo "$output_gpu" | grep 'FINAL_ACCURACY:' | cut -d' ' -f2)
      echo "    GPU -> Treino: ${time_gpu}s, Inferência: ${inf_time_gpu}s, Acurácia: ${acc_gpu}%"
      # Salva no CSV (adiciona inf_time_gpu)
      echo "GPU,$hs,$lr,$ep,$SEED,$time_gpu,$inf_time_gpu,$acc_gpu" >> $OUTPUT_FILE # <<< MODIFICADO

    done
  done
done

echo "Benchmarks concluídos. Resultados salvos em $OUTPUT_FILE"