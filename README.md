# D2SQA 
This is a PyTorch implementation of the paper: D2SQA: A Dynamic Distillation Slow Queries Analysis
Framework via Edge-Cloud Collaboration 

![Overview](./framework.png)

## Environment Preparation
Place D2SQA.yml in the project root directory 


```shell
conda env create --file ./D2SQA.yml  # ./ refers to the current execution directory
```


## Datasets Description & Preprocessing
You can access our data in data directory


## Training
Path descriptions:
- ./src: directory containing source code (e.g., where main.py is located)
- ./model_weights: unified directory for all model weight files
- ./model_weights/combined: subdirectory for combined model weights
- ./model_weights/lora: subdirectory for LoRA adapter weights

```shell
python ./src/main.py \
    -task 'distillation_adapter' \
    -dataset 'tpc_c' \
    -epochs 20 \
    -batch_size 1 \
    -lr 5e-5 \
    -device 'cuda:0' \
    -teacher ./model_weights/combined/tpc_c_llama_8B_teacher.pth \  # Teacher model weights
    -text ./model_weights/combined/tpc_c_llama_8B_text.pth \        # Text module weights
    -cloud2prun tpc_c_llama_8B_cloud2prun \                          # Cloud model used for pruning
    -lora2edge ./model_weights/lora/tpc_c_llama_8B_student_lora/ \   # Student model LoRA weights directory
    -cls2edge ./model_weights/combined/tpc_c_llama_8B_student_cls.pt  # Student model classifier weights
```


## Run run_server.py script (start server)

```shell
python ./src/run_server.py \
    -dataset 'tpc_c' \
    -device 'cuda:0' \
    -text ./model_weights/combined/tpc_c_llama_8B_text.pth \          # Reuse text module weights path
    -sever_model ./model_weights/combined/tpc_c_llama_8B_server.pth \  # Server model weights
    -host '0.0.0.0' \  # Default: allow external devices to access
    -port 5000 \       # Can be adjusted based on actual port usage
    -num_clients 3     # Can be adjusted based on actual number of clients
```

## Run run_client.py script (start client)
Path description: ./model_weights/client: subdirectory for client model weights

```shell
python ./src/run_client.py \
    --client_id 0 \          # Client ID: set according to actual number of clients (e.g., 0, 1, 2...)
    --server_url "http://[server-IP]:5000" \  # Generic format: replace [server-IP] with actual IP (e.g., 192.168.1.100)
    --num_rounds 1 \         # Number of training rounds: adjustable based on requirements
    --num_clients 1 \        # Number of clients being launched: set to 1 for single client
    -device 'cuda:0' \
    -dataset 'tpc_c' \
    -batch_size 1 \
    -lr 1e-7 \
    -client_path ./model_weights/client/tpc_c_llama_8B_distilled_student.pth  # Student model weights with adaptation layer
```
