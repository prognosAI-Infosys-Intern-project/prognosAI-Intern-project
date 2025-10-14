import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import argparse
import os
import joblib

class DataPreprocessor:
    def __init__(self, window_size=30, scaler_type='standard'):
        self.window_size = window_size
        self.scaler_type = scaler_type
        self.scaler = None
        self.feature_cols = None

    def load_data(self, file_path, dataset_type='train'):
        cols = ['engine_id', 'cycle'] + \
               [f'op_setting_{i}' for i in range(1, 4)] + \
               [f'sensor_{i}' for i in range(1, 22)]
        if file_path.endswith('.txt'):
            df = pd.read_csv(file_path, sep=r'\s+', header=None, names=cols)
        else:
            df = pd.read_csv(file_path)
        print(f"Loaded {dataset_type} data: {df.shape}")
        return df

    def calculate_rul(self, df, dataset_type='train', rul_file_path=None):
        df2 = df.copy()
        if dataset_type == 'train':
            max_cycles = df2.groupby('engine_id')['cycle'].max().reset_index()
            max_cycles.columns = ['engine_id', 'max_cycle']
            df2 = df2.merge(max_cycles, on='engine_id')
            df2['RUL'] = df2['max_cycle'] - df2['cycle']
            df2.drop('max_cycle', axis=1, inplace=True)
        elif dataset_type == 'test' and rul_file_path and os.path.exists(rul_file_path):
            rul_true = pd.read_csv(rul_file_path, header=None, names=['RUL_true'])
            rul_true['engine_id'] = rul_true.index + 1
            last_cycles = df2.groupby('engine_id')['cycle'].max().reset_index()
            last_cycles = last_cycles.merge(rul_true, on='engine_id')
            last_cycles.columns = ['engine_id', 'cycle_last', 'RUL_true']
            df2 = df2.merge(last_cycles, on='engine_id')
            df2['RUL'] = df2['RUL_true'] + (df2['cycle_last'] - df2['cycle'])
            df2.drop(['cycle_last', 'RUL_true'], axis=1, inplace=True)
        else:
            df2['RUL'] = 0
        return df2

    def feature_engineering(self, df):
        df3 = df.copy()
        sensor_cols = [c for c in df3.columns if c.startswith('sensor_')]
        const = [c for c in sensor_cols if df3[c].std() == 0]
        df3.drop(columns=const, inplace=True)
        sensor_cols = [c for c in sensor_cols if c not in const]
        for c in sensor_cols:
            grp = df3.groupby('engine_id')[c]
            df3[f'{c}_rollmean5'] = grp.rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
            df3[f'{c}_rollstd5']  = grp.rolling(5, min_periods=1).std().reset_index(level=0, drop=True).fillna(0)
        return df3

    def normalize_features(self, df, dataset_type):
        df4 = df.copy()
        drop_cols = ['engine_id', 'cycle', 'RUL']
        feat_path = os.path.join('processed_data', 'train', 'feature_columns.txt')

        if dataset_type == 'test' and os.path.exists(feat_path):
            with open(feat_path) as f:
                trained = [l.strip() for l in f]
            for c in trained:
                if c not in df4:
                    df4[c] = 0.0
            extra = [c for c in df4.columns if c not in trained + drop_cols]
            df4.drop(columns=extra, inplace=True)
            self.feature_cols = trained
        else:
            self.feature_cols = [c for c in df4.columns if c not in drop_cols]

        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            return df4

        df4[self.feature_cols] = self.scaler.fit_transform(df4[self.feature_cols])
        return df4

    def generate_sequences(self, df):
        seqs, meta = [], []
        df_sorted = df.sort_values(['engine_id', 'cycle'])
        for eid, grp in df_sorted.groupby('engine_id'):
            arr = grp[self.feature_cols].values
            cycles = grp['cycle'].values
            ruls = grp['RUL'].values
            for i in range(self.window_size - 1, len(arr)):
                seqs.append(arr[i-self.window_size+1:i+1])
                meta.append({
                    'engine_id': eid,
                    'cycle': cycles[i],
                    'RUL': ruls[i],
                    'sequence_idx': len(seqs)-1
                })
        return np.array(seqs), pd.DataFrame(meta)

    def save_processed_data(self, seqs, meta_df, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, 'sequences.npy'), seqs)
        meta_df.to_csv(os.path.join(out_dir, 'metadata.csv'), index=False)
        if self.scaler:
            joblib.dump(self.scaler, os.path.join(out_dir, 'scaler.pkl'))
        with open(os.path.join(out_dir, 'feature_columns.txt'), 'w') as f:
            for c in self.feature_cols:
                f.write(c + '\n')
        return

    def process_data(self, raw_path, dataset_type, rul_file, out_dir):
        df = self.load_data(raw_path, dataset_type)
        df = self.calculate_rul(df, dataset_type, rul_file)
        df = self.feature_engineering(df)
        df = self.normalize_features(df, dataset_type)
        seqs, meta_df = self.generate_sequences(df)
        self.save_processed_data(seqs, meta_df, out_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True)
    parser.add_argument('--dataset_type', default='train', choices=['train','test'])
    parser.add_argument('--rul_file', default=None)
    parser.add_argument('--window_size', type=int, default=30)
    parser.add_argument('--scaler_type', default='standard', choices=['standard','minmax','none'])
    args = parser.parse_args()

    base_raw = os.path.join('data', 'raw', args.input_file)
    out_dir = os.path.join('processed_data', args.dataset_type)

    dp = DataPreprocessor(window_size=args.window_size, scaler_type=args.scaler_type)
    dp.process_data(base_raw, args.dataset_type, args.rul_file, out_dir)
    print(f"Processed {args.dataset_type} data saved to {out_dir}")

if __name__ == '__main__':
    # Example direct invocation
    dp = DataPreprocessor(window_size=30, scaler_type='standard')
    dp.process_data(
        raw_path='data/raw/train_FD001.txt',
        dataset_type='train',
        rul_file=None,
        out_dir='processed_data/train'
    )
    dp.process_data(
        raw_path='data/raw/test_FD001.txt',
        dataset_type='test',
        rul_file='data/raw/RUL_FD001.txt',
        out_dir='processed_data/test'
    )
