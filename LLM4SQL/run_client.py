import argparse
import torch
import numpy as np
from client import main_client
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
    parser = argparse.ArgumentParser(description="client")

    # 原有参数
    parser.add_argument("--client_id", type=int, default=0, help="client ID")
    parser.add_argument("--server_url", type=str, default="http://10.82.1.89:5000", help="sever URL")
    parser.add_argument("--num_rounds", type=int, default=1, help="num_rounds")
    parser.add_argument("--num_clients", type=int, default=1, help="num_clients")

    # 合并重复参数（保留并统一参数名，更新默认值）
    parser.add_argument("--batch_size", "-batch_size", type=int, default=1,
                        help="batch size")
    parser.add_argument('-lr', type=float, default=1e-7, help='learning rate')
    parser.add_argument('-device', type=str, default='cuda:0', help='gpu or cpu')
    parser.add_argument('-dataset', type=str, default='tpc_ds', help='dataset name')
    parser.add_argument('-client_path', type=str, help='Edge model with adaptation layer')
    parser.add_argument('-pre_path', type=str, default=pre_path, help='pre path')
    args = parser.parse_args()


    if args.device:
        args.device = args.device
    else:
        if args.cuda and torch.cuda.is_available():
            args.device = f"cuda:{args.cuda_id}"
        else:
            args.device = "cpu"



    main_client(args, args.client_id, args.server_url, args.num_rounds)
