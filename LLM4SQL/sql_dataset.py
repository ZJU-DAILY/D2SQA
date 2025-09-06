from prompt import construct_prompt_lora
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm
import numpy as np
import ast

class TrainDataset(Dataset):
    def __init__(self, args, tokenizer, type, max_length=1024):
        self.ids = []
        self.att_mask = []
        self.labels = []
        self.opt_labels = []

        csv_file = args.pre_path + f"/data/{args.dataset}/{args.dataset}_{type}.csv"
        df = pd.read_csv(csv_file)

        df['opt_label_rate'] = df['opt_label_rate'].apply(self._parse_numeric_list)


        opt_label_array = np.array(df['opt_label_rate'].tolist(), dtype=np.float32)
        opt_label_tensor = torch.tensor(opt_label_array)
        self.opt_label_tensor = opt_label_tensor


        for i in tqdm(range(len(df))):
            sql = df['query'][i]
            multilabel = df['multilabel'][i]
            time = df['duration'][i]
            log = df['plan_json'][i]


            prompt = construct_prompt_lora(sql, time, log)
            output = tokenizer(prompt, return_tensors="pt", max_length=max_length, truncation=True,
                               padding="max_length")
            input_ids = output.input_ids.squeeze(0)
            attention_mask = output.attention_mask.squeeze(0)

            self.ids.append(input_ids)
            self.att_mask.append(attention_mask)


            label_vector = self._parse_multilabel(multilabel)
            self.labels.append(torch.tensor(label_vector, dtype=torch.float32))


            self.opt_labels.append(self.opt_label_tensor[i])

    def _parse_multilabel(self, s):
        if isinstance(s, str):
            s = s.strip("[]").split(",")
        return [int(i) for i in s]

    def _parse_numeric_list(self, s):
        """将字符串形式的列表解析为数值列表"""
        if pd.isna(s):
            return [0.0] * 9
        try:

            return [float(x) for x in ast.literal_eval(s)]
        except:

            print(f"Warning: Unable to parse data: {s}, using default value.")
            return [0.0] * 9
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self.ids[idx], self.att_mask[idx], self.labels[idx], self.opt_labels[idx]



class InferDataset(Dataset):
    def __init__(self, args, tokenizer, max_length=1024):
        """
        init texts, tokenizer, max_length
        """
        self.sqls = []
        self.causes = []
        self.optimizes = []

        self.max_length = max_length
        self.fine_tune = args.fine_tune
        csv_file = args.pre_path + '/data/test.csv'
        df = pd.read_csv(csv_file, encoding='utf-8')
        for i in tqdm(range(len(df))):
            sql = df['SQL'][i]
            cause = df['root_cause'][i]

            if len(sql) > 800:
                sql = sql[:800]

            self.sqls.append(sql)
            self.causes.append(cause)
    def __len__(self):
        return len(self.sqls)

    def __getitem__(self, idx):
        return self.sqls[idx], self.causes[idx]