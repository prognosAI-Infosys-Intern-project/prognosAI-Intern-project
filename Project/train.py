import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Bidirectional, LSTM, Dense, Dropout,
    LayerNormalization, Layer
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os
import joblib

# Custom Self-Attention Layer
class SelfAttention(Layer):
    def __init__(self, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        d_model = input_shape[-1]
        # Query, Key, Value weight matrices
        self.Wq = self.add_weight(
            name="Wq",
            shape=(d_model, d_model),
            initializer="glorot_uniform",
            trainable=True
        )
        self.Wk = self.add_weight(
            name="Wk",
            shape=(d_model, d_model),
            initializer="glorot_uniform",
            trainable=True
        )
        self.Wv = self.add_weight(
            name="Wv",
            shape=(d_model, d_model),
            initializer="glorot_uniform",
            trainable=True
        )
        super(SelfAttention, self).build(input_shape)

    def call(self, inputs):
        # inputs: (batch, timesteps, features)
        Q = tf.matmul(inputs, self.Wq)
        K = tf.matmul(inputs, self.Wk)
        V = tf.matmul(inputs, self.Wv)
        scores = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(tf.cast(tf.shape(K)[-1], tf.float32))
        weights = tf.nn.softmax(scores, axis=-1)
        attended = tf.matmul(weights, V)
        # Pool across time dimension
        return tf.reduce_mean(attended, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])


def load_data_with_scaler(sequence_path, metadata_path, scaler_path, feature_cols_path):
    X = np.load(sequence_path)
    meta = pd.read_csv(metadata_path)
    y = meta['RUL'].values.reshape(-1, 1)

    scaler = joblib.load(scaler_path)
    with open(feature_cols_path, 'r') as f:
        feature_cols = [line.strip() for line in f]

    if X.shape[2] != len(feature_cols):
        raise ValueError(
            f"Expected {len(feature_cols)} features, but got {X.shape[2]}"
        )
    return X, y, scaler, feature_cols


def build_attention_enhanced_lstm(input_shape,
                                  lstm_units=[128, 64],
                                  dropout_rate=0.3,
                                  l2_reg=1e-4,
                                  learning_rate=5e-4):
    inputs = Input(shape=input_shape)
    x = inputs

    # Stacked Bidirectional LSTM layers
    for units in lstm_units:
        x = Bidirectional(
            LSTM(units, return_sequences=True,
                 kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        )(x)
        x = LayerNormalization()(x)
        x = Dropout(dropout_rate)(x)

    # Self-Attention layer
    attn_output = SelfAttention()(x)

    # Dense layers on attention output
    x = Dense(64, activation='relu')(attn_output)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation='linear')(x)

    model = Model(inputs=inputs, outputs=outputs)
    optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model


def main():
    # Paths
    seq_path = os.path.join('processed_data', 'train', 'sequences.npy')
    meta_path = os.path.join('processed_data', 'train', 'metadata.csv')
    scaler_path = os.path.join('processed_data', 'train', 'scaler.pkl')
    feature_cols_path = os.path.join('processed_data', 'train', 'feature_columns.txt')

    # Load data
    X_train, y_train, scaler, feature_cols = load_data_with_scaler(
        seq_path, meta_path, scaler_path, feature_cols_path
    )
    input_shape = (X_train.shape[1], X_train.shape[2])

    # Build model
    model = build_attention_enhanced_lstm(
        input_shape=input_shape,
        lstm_units=[128, 64],
        dropout_rate=0.3,
        l2_reg=1e-4,
        learning_rate=5e-4
    )
    model.summary()

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_mae', patience=10,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_mae', factor=0.3,
                          patience=5, min_lr=1e-6, verbose=1),
        ModelCheckpoint('model/attention_model.keras',
                        save_best_only=True, monitor='val_mae', verbose=1)
    ]

    os.makedirs('model', exist_ok=True)

    # Train
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=2
    )

    # Save final model
    final_model_path = os.path.join('model', 'attention_model_final.keras')
    model.save(final_model_path)
    print(f'Model saved at: {final_model_path}')


if __name__ == '__main__':
    main()
