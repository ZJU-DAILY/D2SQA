import argparse
import torch
from flask import Flask
from sever import app, init_server
import random
import numpy as np
import os
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
if __name__ == "__main__":
    pre_path = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('-text', type=str, help='text model path')
    parser.add_argument('-sever_model', type=str, help='sever model path')
    parser.add_argument('-device', type=str, default='cuda:0', help='gpu or cpu')
    parser.add_argument('-dataset', type=str, help='dataset name')
    parser.add_argument('-pre_path', type=str, default=pre_path, help='pre path')

    parser.add_argument('-host', type=str, default='0.0.0.0', help='Server Host Address')
    parser.add_argument('-port', type=int, default=5000, help='Server port')
    parser.add_argument('-num_clients', type=int, default=3, help='Expected number of clients')
    parser.add_argument('-batch_size', type=int, default=1, help='batch size')
    args = parser.parse_args()


    if torch.cuda.is_available() and 'cuda' in args.device:
        print(f"Use of Equipment: {args.device}")
    else:
        args.device = "cpu"
        print("CUDA is unavailable; using CPU instead.")


    print(f"Initializing federated learning server...")
    init_server(args)

    print(f"Start the server at {args.host}:{args.port}")
    app.run(host=args.host, port=args.port)