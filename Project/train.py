import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os
import joblib


# Load preprocessed data and scaler for consistent feature usage
def load_data_with_scaler(sequence_path, metadata_path, scaler_path, feature_cols_path):
    X = np.load(sequence_path)  # (num_samples, window_size, num_features)
    meta = pd.read_csv(metadata_path)
    y = meta['RUL'].values.reshape(-1, 1)

    # Load scaler and feature columns from training preprocessing
    scaler = joblib.load(scaler_path)
    with open(feature_cols_path, 'r') as f:
        feature_cols = [line.strip() for line in f.readlines()]

    # Confirm feature count matches X's last dimension
    if X.shape[2] != len(feature_cols):
        raise ValueError(f"Feature dimension mismatch! Model expects {len(feature_cols)} features, but sequences have {X.shape[2]} features.")

    return X, y, scaler, feature_cols


def build_model(input_shape, lstm_units=[100, 50], dropout_rate=0.3, l2_reg=1e-4):
    from tensorflow.keras import regularizers

    model = Sequential()
    for idx, units in enumerate(lstm_units):
        return_sequences = (idx < len(lstm_units) - 1)
        if idx == 0:
            model.add(Bidirectional(
                LSTM(units, return_sequences=return_sequences,
                     kernel_regularizer=regularizers.l2(l2_reg)),
                input_shape=input_shape))
        else:
            model.add(Bidirectional(
                LSTM(units, return_sequences=return_sequences,
                     kernel_regularizer=regularizers.l2(l2_reg))
            ))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def main():
    # Paths for preprocessed train data and preprocessing artifacts
    train_seq_path = os.path.join('processed_data', 'train', 'sequences.npy')
    train_meta_path = os.path.join('processed_data', 'train', 'metadata.csv')
    scaler_path = os.path.join('processed_data', 'train', 'scaler.pkl')
    feature_cols_path = os.path.join('processed_data', 'train', 'feature_columns.txt')

    X_train, y_train, scaler, feature_cols = load_data_with_scaler(
        train_seq_path, train_meta_path, scaler_path, feature_cols_path
    )

    print(f"Train sequences shape: {X_train.shape}")
    print(f"Train labels shape: {y_train.shape}")

    input_shape = (X_train.shape[1], X_train.shape[2])

    lstm_units = [100, 75, 50]
    dropout_rate = 0.3
    l2_reg = 1e-4
    model = build_model(input_shape, lstm_units, dropout_rate, l2_reg)
    print(model.summary())

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1),
        ModelCheckpoint('model/best_model.keras', save_best_only=True, monitor='val_loss', verbose=1)
    ]

    os.makedirs('model', exist_ok=True)

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=64,
        callbacks=callbacks,
        verbose=2
    )

    final_model_path = os.path.join('model', 'final_model.keras')
    model.save(final_model_path)
    print(f'Model saved at: {final_model_path}')


if __name__ == '__main__':
    main()
