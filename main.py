import argparse
import os
from train import *
from train_by_embeding import *
from train_distillation_prompt import *
from prun_edge import *
from prun_cloud_model_8B import *
import random
from distillation_adapter import *


pre_path = os.path.dirname(os.path.abspath(__file__))
args = argparse.ArgumentParser()
args.add_argument('-task', type=str, help='train_cloud, train_edge, distillation, prun_cloud, prun_edge, prun_parameter_sensitivity, distillation_adapter')
args.add_argument('-epochs', type=int, default=10, help='epochs')
args.add_argument('-batch_size', type=int, default=1, help='batch size')
args.add_argument('-lr', type=float, default=1e-4, help='learning rate')
args.add_argument('-device', type=str, default='cuda:0', help='gpu or cpu')
args.add_argument('-pre_path', type=str, default=pre_path, help='pre path')
args.add_argument('-dataset', type=str, help='dataset name')
args.add_argument('-teacher', type=str, help='teacher model path')
args.add_argument('-text', type=str, help='text model path')
args.add_argument('-cloud2prun', type=str, help='File name for the pruning cloud model')
args.add_argument('-lora2edge', type=str, help='LORA Weight Path for Edge Models')
args.add_argument('-cls2edge', type=str, help='Classification Head Weight Path for Edge Models')
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
if __name__ == '__main__':
    args.pre_path = pre_path
    args = args.parse_args()
    if args.task == 'train_cloud':
        train_combined_model(args)
    elif args.task == 'train_edge':
        train_llm(args)
    elif args.task == 'distillation':
        train_distillation(args)
    elif args.task == 'prun_cloud':
        cloud_pruning_experiment_8B(args, 0.1, 0.1, 0.15)
    elif args.task == 'prun_edge' or args.task == 'prun_parameter_sensitivity':
        prun_experiment_loop_edge(args)
    elif args.task == 'distillation_adapter':
        train_distillation_client(args)
    else:
        print('?')




