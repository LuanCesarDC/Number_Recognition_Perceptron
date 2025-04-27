#!/bin/bash

# === Configurações ===
declare -a NUM_DENSE_LAYERS=(2 4 8 16)
SEED=42
OUTPUT_FILE="benchmark_depth_results.csv"
GPU_ARCH="sm_86"

# === Compilação (com -lrt) ===
echo "Compilando versão CPU..."
gcc main.c neural_network.c dense_layer.c activation_layer.c hw_number.c nn_math.c -o mnist_cpu_bench -lm -lrt -Wall -Wextra -O2
if [ $? -ne 0 ]; then echo "Falha na compilação CPU!"; exit 1; fi

echo "Compilando versão GPU..."
nvcc main.c neural_network.c dense_layer.c activation_layer.c hw_number.c nn_math_cuda.cu -o mnist_gpu_bench -DUSE_CUDA -arch=${GPU_ARCH} -lm -lrt -Xcompiler "-Wall -Wextra -O2"
if [ $? -ne 0 ]; then echo "Falha na compilação GPU!"; exit 1; fi

# === Execução dos Testes ===
echo "Iniciando Benchmarks de Profundidade..."
# Cria cabeçalho do CSV (modificado)
echo "Backend,NumDenseLayers,Seed,TrainingTime(s),InferenceTime(s),Accuracy(%)" > $OUTPUT_FILE

# Loop pelas configurações de profundidade
for ndl in "${NUM_DENSE_LAYERS[@]}"; do
  echo "--- Executando: NumDenseLayers=$ndl ---"

  # --- Run CPU ---
  echo "  Rodando CPU..."
  # Executa passando o número de camadas densas e a semente
  output_cpu=$(./mnist_cpu_bench $ndl $SEED)
  # Extrai tempo de treino, tempo de inferência e acurácia
  time_cpu=$(echo "$output_cpu" | grep 'TRAINING_TIME:' | cut -d' ' -f2)
  inf_time_cpu=$(echo "$output_cpu" | grep 'INFERENCE_TIME:' | cut -d' ' -f2)
  acc_cpu=$(echo "$output_cpu" | grep 'FINAL_ACCURACY:' | cut -d' ' -f2)
  echo "    CPU -> Treino: ${time_cpu}s, Inferência: ${inf_time_cpu}s, Acurácia: ${acc_cpu}%"
  # Salva no CSV (modificado)
  echo "CPU,$ndl,$SEED,$time_cpu,$inf_time_cpu,$acc_cpu" >> $OUTPUT_FILE

  # --- Run GPU ---
  echo "  Rodando GPU..."
  # Executa passando o número de camadas densas e a semente
  output_gpu=$(./mnist_gpu_bench $ndl $SEED)
  # Extrai tempo de treino, tempo de inferência e acurácia
  time_gpu=$(echo "$output_gpu" | grep 'TRAINING_TIME:' | cut -d' ' -f2)
  inf_time_gpu=$(echo "$output_gpu" | grep 'INFERENCE_TIME:' | cut -d' ' -f2)
  acc_gpu=$(echo "$output_gpu" | grep 'FINAL_ACCURACY:' | cut -d' ' -f2)
  echo "    GPU -> Treino: ${time_gpu}s, Inferência: ${inf_time_gpu}s, Acurácia: ${acc_gpu}%"
  # Salva no CSV (modificado)
  echo "GPU,$ndl,$SEED,$time_gpu,$inf_time_gpu,$acc_gpu" >> $OUTPUT_FILE

done

echo "Benchmarks concluídos. Resultados salvos em $OUTPUT_FILE"