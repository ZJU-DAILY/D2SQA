from prompt import construct_prompt_lora
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm
import numpy as np
import ast
from Embedding import PlanFeatureExtractor
from nltk.tokenize import word_tokenize


class CombinedDataset(Dataset):
    def __init__(self, args, type, max_length=1024):
        self.prompts = []
        self.labels = []
        self.opt_labels = []
        self.graph_data = []
        self.time_series = []

        self.feature_extractor = PlanFeatureExtractor()

        csv_file = args.pre_path + f"/data/{args.dataset}/{args.dataset}_{type}.csv"
        df = pd.read_csv(csv_file)
        df['opt_label_rate'] = df['opt_label_rate'].apply(self._parse_numeric_list)

        opt_label_array = np.array(df['opt_label_rate'].tolist(), dtype=np.float32)
        self.opt_label_tensor = torch.tensor(opt_label_array)

        for i in tqdm(range(len(df)), desc=f"Loading {type} data"):
            sql = df['query'][i]
            multilabel = df['multilabel'][i]
            time = df['duration'][i]
            log = df['plan_json'][i]

            prompt = construct_prompt_lora(sql, time, log)
            self.prompts.append(prompt)

            parsed_plan = self.feature_extractor.parse_plan_string(log)
            if parsed_plan and 'Plan' in parsed_plan[0]:
                plan_root = parsed_plan[0]['Plan']
                graph = self.feature_extractor.build_graph(plan_root)
                self.graph_data.append(graph)
            else:
                print("Warning: failed to parse plan, using default graph")
                default_graph = {
                    "x": self.feature_extractor.get_default_features().unsqueeze(0),
                    "edge_index": torch.tensor([], dtype=torch.long)
                }
                self.graph_data.append(default_graph)

            if 'timeseries' in df.columns:
                ts_data = df['timeseries'][i]
                ts_tensor = self._parse_time_series(ts_data)
                self.time_series.append(ts_tensor)
            else:
                self.time_series.append(torch.zeros(9, 7))

            label_vector = self._parse_multilabel(multilabel)
            self.labels.append(torch.tensor(label_vector, dtype=torch.float32))
            self.opt_labels.append(self.opt_label_tensor[i])

    def _parse_time_series(self, s):
        if pd.isna(s) if isinstance(s, (float, int, str)) else (isinstance(s, np.ndarray) and np.isnan(s).all()):
            print(f"Warning: time-series data contains missing values or is all NaN - Type: {type(s)}, Value: {s}")
            return torch.zeros((7, 9, 1), dtype=torch.float32)

        try:
            if isinstance(s, str):
                try:
                    s = ast.literal_eval(s)
                except Exception:
                    raise ValueError(f"Unable to parse time-series string: {s}")

            data_array = np.array(s, dtype=np.float32)

            if data_array.ndim == 1:
                if len(data_array) != 63:
                    raise ValueError(f"1-D array length must be 63, got {len(data_array)}")
                data_array = data_array.reshape(7, 9)

            if data_array.shape != (7, 9):
                raise ValueError(f"Time-series shape must be (7, 9), got {data_array.shape}")

            return torch.tensor(data_array.reshape(7, 9, 1), dtype=torch.float32)

        except Exception as e:
            print(f"Warning: time-series parsing failed - {str(e)}, using default")
            return torch.zeros((7, 9, 1), dtype=torch.float32)

    def _parse_multilabel(self, s):
        if isinstance(s, str):
            s = s.strip("[]").split(",")
        return [int(i) for i in s]

    def _parse_numeric_list(self, s):
        if pd.isna(s):
            return [0.0] * 9
        try:
            return [float(x) for x in ast.literal_eval(s)]
        except Exception:
            print(f"Warning: unable to parse data: {s}, using default")
            return [0.0] * 9

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return (
            self.prompts[idx],
            self.labels[idx],
            self.opt_labels[idx],
            self.graph_data[idx],
            self.time_series[idx]
        )
