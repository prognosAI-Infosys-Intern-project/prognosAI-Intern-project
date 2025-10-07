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
        column_names = ['engine_id', 'cycle'] + \
                       [f'op_setting_{i}' for i in range(1, 4)] + \
                       [f'sensor_{i}' for i in range(1, 22)]
        if file_path.endswith('.txt'):
            df = pd.read_csv(file_path, sep=r'\s+', header=None, names=column_names)
        else:
            df = pd.read_csv(file_path)
        print(f"Loaded {dataset_type} data: {df.shape}")
        return df

    def calculate_rul(self, df, dataset_type='train', rul_file_path=None):
        df_rul = df.copy()
        if dataset_type == 'train':
            max_cycles = df_rul.groupby('engine_id')['cycle'].max().reset_index()
            max_cycles.columns = ['engine_id', 'max_cycle']
            df_rul = df_rul.merge(max_cycles, on='engine_id', how='left')
            df_rul['RUL'] = df_rul['max_cycle'] - df_rul['cycle']
            df_rul.drop('max_cycle', axis=1, inplace=True)
        elif dataset_type == 'test' and rul_file_path and os.path.exists(rul_file_path):
            rul_true = pd.read_csv(rul_file_path, header=None, names=['RUL_true'])
            rul_true['engine_id'] = rul_true.index + 1
            last_cycles = df_rul.groupby('engine_id')['cycle'].max().reset_index()
            last_cycles = last_cycles.merge(rul_true, on='engine_id', how='left')
            df_rul = df_rul.merge(last_cycles[['engine_id', 'cycle', 'RUL_true']],
                                  on='engine_id', how='left', suffixes=('', '_last'))
            df_rul['RUL'] = df_rul['RUL_true'] + (df_rul['cycle_last'] - df_rul['cycle'])
            df_rul.drop(['RUL_true', 'cycle_last'], axis=1, inplace=True)
        else:
            df_rul['RUL'] = 0
        return df_rul

    def feature_engineering(self, df):
        df_features = df.copy()
        sensor_cols = [col for col in df_features.columns if 'sensor_' in col]
        constant_sensors = [col for col in sensor_cols if df_features[col].std() == 0]
        if constant_sensors:
            df_features.drop(constant_sensors, axis=1, inplace=True)
            sensor_cols = [col for col in sensor_cols if col not in constant_sensors]

        for col in sensor_cols:
            df_features[f'{col}_rollmean5'] = df_features.groupby('engine_id')[col].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True)
            df_features[f'{col}_rollstd5'] = df_features.groupby('engine_id')[col].rolling(window=5, min_periods=1).std().reset_index(level=0, drop=True).fillna(0)

        return df_features

    def normalize_features(self, df):
        if self.scaler_type == 'none':
            return df
        df_norm = df.copy()

        exclude_cols = ['engine_id', 'cycle', 'RUL']

        # For test dataset: Align features exactly as training features if feature_columns.txt exists
        feature_cols_path = os.path.join('processed_data', 'train', 'feature_columns.txt')
        if os.path.exists(feature_cols_path):
            with open(feature_cols_path, 'r') as f:
                trained_features = [line.strip() for line in f.readlines()]
            # Add missing columns with zeros
            for col in trained_features:
                if col not in df_norm.columns:
                    df_norm[col] = 0.0
            # Remove extra columns not in trained features
            extra_cols = [col for col in df_norm.columns if col not in trained_features and col not in exclude_cols]
            if extra_cols:
                df_norm.drop(extra_cols, axis=1, inplace=True)
            # Reorder columns
            self.feature_cols = trained_features
        else:
            # Training flow: take all except exclude
            self.feature_cols = [col for col in df_norm.columns if col not in exclude_cols]

        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")

        df_norm[self.feature_cols] = self.scaler.fit_transform(df_norm[self.feature_cols])
        return df_norm

    def generate_sequences(self, df):
        sequences = []
        metadata = []
        df_sorted = df.sort_values(['engine_id', 'cycle']).reset_index(drop=True)
        for engine_id in df_sorted['engine_id'].unique():
            engine_data = df_sorted[df_sorted['engine_id'] == engine_id]
            features = engine_data[self.feature_cols].values
            cycles = engine_data['cycle'].values
            ruls = engine_data['RUL'].values
            for i in range(self.window_size - 1, len(engine_data)):
                seq = features[i - self.window_size + 1:i + 1]
                sequences.append(seq)
                metadata.append({'engine_id': engine_id, 'cycle': cycles[i], 'RUL': ruls[i], 'sequence_idx': len(sequences) - 1})
        sequences = np.array(sequences)
        metadata_df = pd.DataFrame(metadata)
        return sequences, metadata_df

    def save_processed_data(self, sequences, metadata_df, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        seq_path = os.path.join(output_dir, 'sequences.npy')
        meta_path = os.path.join(output_dir, 'metadata.csv')
        np.save(seq_path, sequences)
        metadata_df.to_csv(meta_path, index=False)
        if self.scaler:
            scaler_path = os.path.join(output_dir, 'scaler.pkl')
            joblib.dump(self.scaler, scaler_path)
        feature_path = os.path.join(output_dir, 'feature_columns.txt')
        with open(feature_path, 'w') as f:
            for col in self.feature_cols:
                f.write(col + '\n')
        return seq_path, meta_path

    def process_data(self, input_file, dataset_type='train', rul_file=None, output_dir='processed_data'):
        df = self.load_data(input_file, dataset_type)
        df = self.calculate_rul(df, dataset_type, rul_file)
        df = self.feature_engineering(df)
        df = self.normalize_features(df)
        sequences, metadata_df = self.generate_sequences(df)
        return self.save_processed_data(sequences, metadata_df, output_dir)


def main():
    parser = argparse.ArgumentParser(description='CMAPSS Data Preprocessing')
    parser.add_argument('--input_file', type=str, required=True, help='Raw data file path')
    parser.add_argument('--dataset_type', type=str, default='train', choices=['train','test'])
    parser.add_argument('--rul_file', type=str, default=None, help='RUL file path for test data')
    parser.add_argument('--window_size', type=int, default=30, help='Window size for sequence generation')
    parser.add_argument('--scaler_type', type=str, default='standard', choices=['standard','minmax','none'])
    parser.add_argument('--output_dir', type=str, default='processed_data', help='Output directory')
    args = parser.parse_args()

    # Adjust input file path to read from 'data/raw/'
    input_filepath = os.path.join('data', 'raw', args.input_file)

    # Adjust output directory to save into 'train/' or 'test/' folders
    out_dir = os.path.join(args.output_dir, args.dataset_type)

    preprocessor = DataPreprocessor(window_size=args.window_size, scaler_type=args.scaler_type)
    seq_path, meta_path = preprocessor.process_data(input_filepath, args.dataset_type, args.rul_file, out_dir)
    print(f"Sequences saved to: {seq_path}")
    print(f"Metadata saved to: {meta_path}")


if __name__ == '__main__':
    class Args:
        window_size = 30
        scaler_type = 'standard'
        output_dir = 'processed_data'

    preprocessor = DataPreprocessor(window_size=Args.window_size, scaler_type=Args.scaler_type)

    # Process training data
    train_input = os.path.join('data', 'raw', 'train_FD001.txt')
    train_output = os.path.join(Args.output_dir, 'train')
    train_seq_path, train_meta_path = preprocessor.process_data(train_input, 'train', None, train_output)
    print(f"Train sequences saved to: {train_seq_path}")
    print(f"Train metadata saved to: {train_meta_path}")

    # Process test data
    test_input = os.path.join('data', 'raw', 'test_FD001.txt')
    test_rul_file = os.path.join('data', 'raw', 'RUL_FD001.txt')
    test_output = os.path.join(Args.output_dir, 'test')
    test_seq_path, test_meta_path = preprocessor.process_data(test_input, 'test', test_rul_file, test_output)
    print(f"Test sequences saved to: {test_seq_path}")
    print(f"Test metadata saved to: {test_meta_path}")
