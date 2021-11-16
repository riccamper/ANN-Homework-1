# Import needed libraries
from sklearn.model_selection import KFold
import tensorflow as tf
import numpy as np
import os
import random
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Test the Keras version
tfk = tf.keras
tfkl = tf.keras.layers
print(tf.__version__)

# Random seed for reproducibility
seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)

# Load wine file
wine = pd.read_csv(r'labs\winequality.csv', sep=';')
print(wine.info())
wine.head()
wine.describe()

# ----------------------------------------------------------

# Create train set, test set, train val set
X_train_val, X_test, y_train_val, y_test = train_test_split(
    wine[wine.columns[:-1]],
    pd.DataFrame(wine['quality'], columns=['quality']),
    test_size=0.1,
    random_state=seed,
    stratify=pd.DataFrame(wine['quality'], columns=['quality'])
)
print(X_train_val.shape, y_train_val.shape)
print(X_test.shape, y_test.shape)

# Inspect the target
plt.figure(figsize=(15, 5))
sns.histplot(data=y_train_val, x='quality', kde=True)
plt.show()

# Normalize both features and target
max_df = X_train_val.max()
min_df = X_train_val.min()
max_t = y_train_val.max()
min_t = y_train_val.min()
X_train_val = (X_train_val - min_df)/(max_df - min_df)
y_train_val = (y_train_val - min_t)/(max_t - min_t)
print('Wine dataset shape', X_train_val.shape)
print('Target shape', y_train_val.shape)
X_train_val.describe()

# Inspect the target after normalization
plt.figure(figsize=(15, 5))
sns.histplot(data=y_train_val, x='quality', kde=True)
plt.show()

# Normalize the test set with the same parameters of training set
X_test = (X_test - min_df)/(max_df - min_df)
y_test = (y_test - min_t)/(max_t - min_t)

# ----------------------------------------------------------

# Hold out
input_shape = X_train_val.shape[1:]
batch_size = 256
epochs = 1000

def monitor(histories, names, colors, early_stopping=1):
    assert len(histories) == len(names)
    assert len(histories) == len(colors)
    plt.figure(figsize=(15, 6))
    for idx in range(len(histories)):
        plt.plot(histories[idx]['mse'][:-early_stopping], label=names[idx] +
                 ' Training', alpha=.4, color=colors[idx], linestyle='--')
        plt.plot(histories[idx]['val_mse'][:-early_stopping],
                 label=names[idx]+' Validation', alpha=.8, color=colors[idx])
    plt.ylim(0.0075, 0.02)
    plt.title('Mean Squared Error')
    plt.legend(bbox_to_anchor=(1, 1))
    plt.grid(alpha=.3)
    plt.show()

def plot_residuals(model, X_, y_):
    X_['sort'] = y_
    X_ = X_.sort_values(by=['sort'])
    y_ = np.expand_dims(X_['sort'], 1)
    X_.drop(['sort'], axis=1, inplace=True)

    y_pred = model.predict(X_)
    MSE = mean_squared_error(y_, y_pred)

    print('Mean Squared Error (MSE): %.4f' % MSE)

    mpl.rcParams.update(mpl.rcParamsDefault)
    sns.set(font_scale=1.1, style=None, palette='Set1')
    plt.figure(figsize=(15, 5))
    plt.scatter(np.arange(len(y_pred)), y_pred,
                label='Prediction', color='#1f77b4')
    plt.scatter(np.arange(len(y_)), y_, label='True', color='#d62728')

    for i in range(len(y_)):
        if(y_[i] >= y_pred[i]):
            plt.vlines(i, y_pred[i], y_[i], alpha=.5)
        else:
            plt.vlines(i, y_[i], y_pred[i], alpha=.5)

    plt.legend()
    plt.grid(alpha=.3)
    plt.show()

histories = []
names = []
colors = []
val_scores = []
test_scores = []

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=len(X_test),
    random_state=seed,
    stratify=y_train_val
)
print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)

# ----------------------------------------------------------

# Default model
def build_default_model(input_shape):

    # Build the neural network layer by layer
    input_layer = tfkl.Input(shape=input_shape, name='Input')
    hidden_layer1 = tfkl.Dense(units=256, activation='relu', name='Hidden1',
                               kernel_initializer=tfk.initializers.GlorotUniform(seed=seed))(input_layer)
    hidden_layer2 = tfkl.Dense(units=128, activation='relu', name='Hidden2',
                               kernel_initializer=tfk.initializers.GlorotUniform(seed=seed))(hidden_layer1)
    hidden_layer3 = tfkl.Dense(units=64, activation='relu', name='Hidden3',
                               kernel_initializer=tfk.initializers.GlorotUniform(seed=seed))(hidden_layer2)
    output_layer = tfkl.Dense(units=1, activation='linear', name='Output',
                              kernel_initializer=tfk.initializers.GlorotUniform(seed=seed))(hidden_layer3)

    # Connect input and output through the Model class
    model = tfk.Model(inputs=input_layer,
                      outputs=output_layer, name='default_model')

    # Compile the model
    learning_rate = 1e-3
    opt = tfk.optimizers.Adam(learning_rate)
    loss = tfk.losses.MeanSquaredError()
    mtr = ['mse']
    model.compile(loss=loss, optimizer=opt, metrics=mtr)

    # Return the model
    return model


default_model = build_default_model(input_shape)
default_model.summary()
# tfk.utils.plot_model(default_model)

# Train
default_history = default_model.fit(
    x=X_train,
    y=y_train,
    validation_data=(X_val, y_val),
    batch_size=batch_size,
    epochs=epochs
).history

plt.figure(figsize=(15, 5))
plt.plot(default_history['mse'], label='Training', alpha=.8, color='#ff7f0e')
plt.plot(default_history['val_mse'],
         label='Validation', alpha=.8, color='#4D61E2')
plt.ylim(0, 0.025)
plt.title('Mean Squared Error')
plt.legend(loc='upper right')
plt.grid(alpha=.3)
plt.show()

print('Train Performance')
plot_residuals(default_model, X_train.copy(), y_train.copy())
print('Validation Performance')
plot_residuals(default_model, X_val.copy(), y_val.copy())

# ----------------------------------------------------------

# Early stopping
patience = 150
early_stopping = tfk.callbacks.EarlyStopping(
    monitor='val_mse', mode='min', patience=patience, restore_best_weights=True)

earlystopping_model = build_default_model(input_shape)
earlystopping_model.summary()
# tfk.utils.plot_model(earlystopping_model)

earlystopping_history = earlystopping_model.fit(
    x=X_train,
    y=y_train,
    validation_data=(X_val, y_val),
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[early_stopping]
).history

plt.figure(figsize=(15, 5))
plt.plot(default_history['mse'], label='Training', alpha=.3, color='#ff7f0e')
plt.plot(default_history['val_mse'],
         label='Validation', alpha=.3, color='#4D61E2')
plt.plot(earlystopping_history['mse'],
         label='Training (early stopping)', alpha=.8, color='#ff7f0e')
plt.plot(earlystopping_history['val_mse'],
         label='Validation (early stopping)', alpha=.8, color='#4D61E2')
plt.ylim(0, 0.03)
plt.title('Mean Squared Error')
plt.legend(loc='upper right')
plt.grid(alpha=.3)
plt.show()

print('Train Performance')
plot_residuals(earlystopping_model, X_train.copy(), y_train.copy())
print('Validation Performance')
plot_residuals(earlystopping_model, X_val.copy(), y_val.copy())

# Store results
val_scores.append(mean_squared_error(
    y_val, earlystopping_model.predict(X_val)))
print('Validation MSE %.4f' % val_scores[0])
test_scores.append(mean_squared_error(
    y_test, earlystopping_model.predict(X_test)))
histories.append(earlystopping_history)
names.append('Default')
colors.append('#ff7f0e')
monitor(histories, names, colors, patience)

# Save, delete and load a Keras model
earlystopping_model.save('labs\WineTrain')
#del earlystopping_model
#earlystopping_model = tfk.models.load_model('DefaultModel')
#print('Validation MSE: %.4f' % mean_squared_error(y_val, earlystopping_model.predict(X_val)))

# ----------------------------------------------------------

# Weight decay (Regularization techniques)
def build_l2_model(input_shape):

    # Build the neural network layer by layer
    input_layer = tfkl.Input(shape=input_shape, name='Input')
    hidden_layer1 = tfkl.Dense(units=256, activation='relu', name='Hidden1',
                               kernel_initializer=tfk.initializers.GlorotUniform(
                                   seed=seed),
                               kernel_regularizer=tf.keras.regularizers.l2(1e-5))(input_layer)
    hidden_layer2 = tfkl.Dense(units=128, activation='relu', name='Hidden2',
                               kernel_initializer=tfk.initializers.GlorotUniform(
                                   seed=seed),
                               kernel_regularizer=tf.keras.regularizers.l2(1e-5))(hidden_layer1)
    hidden_layer3 = tfkl.Dense(units=64, activation='relu', name='Hidden3',
                               kernel_initializer=tfk.initializers.GlorotUniform(
                                   seed=seed),
                               kernel_regularizer=tf.keras.regularizers.l2(1e-5))(hidden_layer2)
    output_layer = tfkl.Dense(units=1, activation='linear', name='Output',
                              kernel_initializer=tfk.initializers.GlorotUniform(
                                  seed=seed),
                              kernel_regularizer=tf.keras.regularizers.l2(1e-5))(hidden_layer3)

    # Connect input and output through the Model class
    model = tfk.Model(inputs=input_layer,
                      outputs=output_layer, name='l2_model')

    # Compile the model
    learning_rate = 1e-3
    opt = tfk.optimizers.Adam(learning_rate)
    loss = tfk.losses.MeanSquaredError()
    mtr = ['mse']
    model.compile(loss=loss, optimizer=opt, metrics=mtr)

    # Return the model
    return model

l2_model = build_l2_model(input_shape)
l2_model.summary()
# tfk.utils.plot_model(l2_model)

history_l2 = l2_model.fit(
    x=X_train,
    y=y_train,
    validation_data=(X_val, y_val),
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[early_stopping]
).history

# Store results
val_scores.append(mean_squared_error(y_val, l2_model.predict(X_val)))
print('Validation MSE %.4f' % val_scores[1])
test_scores.append(mean_squared_error(y_test, l2_model.predict(X_test)))
histories.append(history_l2)
names.append('Weight Decay')
colors.append('#4D61E2')
monitor(histories, names, colors, patience)

# ----------------------------------------------------------

# Dropout (Regularization techniques)
def build_dropout_model(input_shape):

    # Build the neural network layer by layer
    input_layer = tfkl.Input(shape=input_shape, name='Input')
    hidden_layer1 = tfkl.Dense(units=256, activation='relu', name='Hidden1',
                               kernel_initializer=tfk.initializers.GlorotUniform(seed=seed))(input_layer)
    hidden_layer1 = tfkl.Dropout(0.4, seed=seed)(hidden_layer1)
    hidden_layer2 = tfkl.Dense(units=128, activation='relu', name='Hidden2',
                               kernel_initializer=tfk.initializers.GlorotUniform(seed=seed))(hidden_layer1)
    hidden_layer2 = tfkl.Dropout(0.4, seed=seed)(hidden_layer2)
    hidden_layer3 = tfkl.Dense(units=64, activation='relu', name='Hidden3',
                               kernel_initializer=tfk.initializers.GlorotUniform(seed=seed))(hidden_layer2)
    hidden_layer3 = tfkl.Dropout(0.4, seed=seed)(hidden_layer3)
    output_layer = tfkl.Dense(units=1, activation='linear', name='Output',
                              kernel_initializer=tfk.initializers.GlorotUniform(
                                  seed=seed),
                              kernel_regularizer=tf.keras.regularizers.l2(1e-5))(hidden_layer3)

    # Connect input and output through the Model class
    model = tfk.Model(inputs=input_layer,
                      outputs=output_layer, name='dropout_model')

    # Compile the model
    learning_rate = 1e-3
    opt = tfk.optimizers.Adam(learning_rate)
    loss = tfk.losses.MeanSquaredError()
    mtr = ['mse']
    model.compile(loss=loss, optimizer=opt, metrics=mtr)

    # Return the model
    return model

dropout_model = build_dropout_model(input_shape)
dropout_model.summary()
# tfk.utils.plot_model(dropout_model)

history_dropout = dropout_model.fit(
    x=X_train,
    y=y_train,
    validation_data=(X_val, y_val),
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[early_stopping]
).history

# Store results
val_scores.append(mean_squared_error(y_val, dropout_model.predict(X_val)))
print('Validation MSE %.4f' % val_scores[2])
test_scores.append(mean_squared_error(y_test, dropout_model.predict(X_test)))
histories.append(history_dropout)
names.append('Dropout')
colors.append('#7DD667')
monitor(histories, names, colors, patience)

# ----------------------------------------------------------

# Dropout + l2-norm (Regularization techniques)
def build_dropout_l2_model(input_shape):

    # Build the neural network layer by layer
    input_layer = tfkl.Input(shape=input_shape, name='Input')
    hidden_layer1 = tfkl.Dense(units=256, activation='relu', name='Hidden1',
                               kernel_initializer=tfk.initializers.GlorotUniform(
                                   seed=seed),
                               kernel_regularizer=tf.keras.regularizers.l2(2e-6))(input_layer)
    hidden_layer1 = tfkl.Dropout(0.3, seed=seed)(hidden_layer1)
    hidden_layer2 = tfkl.Dense(units=128, activation='relu', name='Hidden2',
                               kernel_initializer=tfk.initializers.GlorotUniform(
                                   seed=seed),
                               kernel_regularizer=tf.keras.regularizers.l2(2e-6))(hidden_layer1)
    hidden_layer2 = tfkl.Dropout(0.3, seed=seed)(hidden_layer2)
    hidden_layer3 = tfkl.Dense(units=64, activation='relu', name='Hidden3',
                               kernel_initializer=tfk.initializers.GlorotUniform(
                                   seed=seed),
                               kernel_regularizer=tf.keras.regularizers.l2(2e-6))(hidden_layer2)
    hidden_layer3 = tfkl.Dropout(0.3, seed=seed)(hidden_layer3)
    output_layer = tfkl.Dense(units=1, activation='linear', name='Output',
                              kernel_initializer=tfk.initializers.GlorotUniform(
                                  seed=seed),
                              kernel_regularizer=tf.keras.regularizers.l2(2e-6))(hidden_layer3)

    # Connect input and output through the Model class
    model = tfk.Model(inputs=input_layer, outputs=output_layer,
                      name='dropout_l2_model')

    # Compile the model
    learning_rate = 1e-3
    opt = tfk.optimizers.Adam(learning_rate)
    loss = tfk.losses.MeanSquaredError()
    mtr = ['mse']
    model.compile(loss=loss, optimizer=opt, metrics=mtr)

    # Return the model
    return model

dropoutl2_model = build_dropout_l2_model(input_shape)
dropoutl2_model.summary()
# tfk.utils.plot_model(dropoutl2_model)

history_dropoutl2 = dropoutl2_model.fit(
    x=X_train,
    y=y_train,
    validation_data=(X_val, y_val),
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[early_stopping]
).history

# Store results
val_scores.append(mean_squared_error(y_val, dropoutl2_model.predict(X_val)))
print('Validation MSE %.4f' % val_scores[3])
test_scores.append(mean_squared_error(y_test, dropoutl2_model.predict(X_test)))
histories.append(history_dropoutl2)
names.append('Dropout + L2')
colors.append('#B951D0')
monitor(histories, names, colors, patience)

plt.figure(figsize=(15, 6))
plt.plot(default_history['val_mse'], alpha=.3, color='#ff7f0e')
plt.plot(earlystopping_history['val_mse'][:-patience],
         alpha=.8, color='#ff7f0e', label='Default')
plt.plot(history_l2['val_mse'], alpha=.3, color='#4D61E2')
plt.plot(history_l2['val_mse'][:-patience], alpha=.8,
         color='#4D61E2', label='Weight Decay')
plt.plot(history_dropout['val_mse'], alpha=.3, color='#7DD667')
plt.plot(history_dropout['val_mse'][:-patience],
         alpha=.8, color='#7DD667', label='Dropout')
plt.plot(history_dropoutl2['val_mse'], alpha=.3, color='#B951D0')
plt.plot(history_dropoutl2['val_mse'][:-patience], alpha=.8,
         color='#B951D0', label='Dropout + Weight Decay')
plt.ylim(0.0115, 0.025)
plt.title('Mean Squared Error')
plt.legend(loc='upper right')
plt.grid(alpha=.3)
plt.show()

plt.figure(figsize=(15, 6))
plt.bar(names, val_scores, color=colors, alpha=.8)
plt.ylim(0, .015)
plt.title('Validation MSE')
plt.grid(alpha=.3, axis='y')
plt.show()

plt.figure(figsize=(15, 6))
plt.bar(names, test_scores, color=colors, alpha=.8)
plt.ylim(0.01, .016)
plt.title('Validation MSE')
plt.grid(alpha=.3, axis='y')
plt.show()

print('Train Performance with Best Model')
plot_residuals(dropout_model, X_train.copy(), y_train.copy())
print('Validation Performance with Best Model')
plot_residuals(dropout_model, X_val.copy(), y_val.copy())
print('Test Performance with Best Model')
plot_residuals(dropout_model, X_test.copy(), y_test.copy())

# ----------------------------------------------------------

# K-Fold
num_folds = 10
histories = []
scores = []

kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)

for fold_idx, (train_idx, valid_idx) in enumerate(kfold.split(X_train_val, y_train_val)):
    print("Starting training on fold num: {}".format(fold_idx+1))
    model = build_dropout_model(input_shape)

    history = model.fit(
        x=X_train_val.iloc[train_idx],
        y=y_train_val.iloc[train_idx],
        validation_data=(
            X_train_val.iloc[valid_idx], y_train_val.iloc[valid_idx]),
        batch_size=batch_size,
        epochs=100,
        callbacks=[early_stopping]
    ).history

    score = model.evaluate(
        X_train_val.iloc[valid_idx], y_train_val.iloc[valid_idx])
    scores.append(score[1])
    histories.append(history)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

print("MSE")
print("Mean: {}; STD: {}".format(np.mean(scores).round(4), np.std(scores).round(4)))

plt.figure(figsize=(15,6))
for fold_idx in range(num_folds):
  plt.plot(histories[fold_idx]['val_mse'], color=colors[fold_idx], label='Fold N°{}'.format(fold_idx+1))
  plt.ylim(0.011, 0.03)
  plt.title('Mean Squared Error')
  plt.legend(loc='upper right')
  plt.grid(alpha=.3)
plt.show()
