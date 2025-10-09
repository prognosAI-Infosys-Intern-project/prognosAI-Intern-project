import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense, Dropout, BatchNormalization, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os
import joblib

# Custom Attention Layer
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], 1),
            initializer='random_normal',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(input_shape[1], 1),
            initializer='zeros',
            trainable=True
        )
        super(Attention, self).build(input_shape)

    def call(self, inputs):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(inputs, self.W) + self.b)
        e = tf.keras.backend.squeeze(e, axis=-1)  # (batch_size, timesteps)
        alpha = tf.keras.backend.softmax(e)       # attention weights
        alpha_expanded = tf.keras.backend.expand_dims(alpha, axis=-1)  # (batch_size, timesteps, 1)
        context = inputs * alpha_expanded
        context = tf.keras.backend.sum(context, axis=1)  # context vector
        return context

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

def build_stacked_bidir_lstm_attention(input_shape,
                                       lstm_units=[128, 64, 32],
                                       dropout_rate=0.5,
                                       l2_reg=2e-3,
                                       learning_rate=1e-4):
    inputs = Input(shape=input_shape)
    x = inputs
    for units in lstm_units:
        x = Bidirectional(
            LSTM(units,
                 return_sequences=True,
                 kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        )(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)

    context = Attention()(x)

    outputs = Dense(1, activation='linear')(context)

    model = Model(inputs=inputs, outputs=outputs)
    optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

def main():
    train_seq_path = os.path.join('processed_data', 'train', 'sequences.npy')
    train_meta_path = os.path.join('processed_data', 'train', 'metadata.csv')
    scaler_path = os.path.join('processed_data', 'train', 'scaler.pkl')
    feature_cols_path = os.path.join('processed_data', 'train', 'feature_columns.txt')

    X_train, y_train, scaler, feature_cols = load_data_with_scaler(
        train_seq_path, train_meta_path, scaler_path, feature_cols_path
    )

    input_shape = (X_train.shape[1], X_train.shape[2])

    model = build_stacked_bidir_lstm_attention(
        input_shape=input_shape,
        lstm_units=[128, 64, 32],
        dropout_rate=0.5,
        l2_reg=2e-3,
        learning_rate=1e-4
    )
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_mae', patience=12, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_mae', factor=0.2, patience=6, min_lr=1e-6, verbose=1),
        ModelCheckpoint('model/best_model.keras', save_best_only=True, monitor='val_mae', verbose=1)
    ]

    os.makedirs('model', exist_ok=True)

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=2
    )

    final_model_path = os.path.join('model', 'final_model.keras')
    model.save(final_model_path)
    print(f'Model saved at: {final_model_path}')

if __name__ == '__main__':
    main()
