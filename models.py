# models.py
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2

import numpy as np
def train_mlp(X_train, y_train):
    hidden_layer_sizes = [(16,), (8, 4)]
    activation = ['relu', 'tanh']
    solver = ['adam', 'sgd']
    alpha = [0.01, 0.1]
    learning_rate = ['constant', 'adaptive']

    mlp_params = {
        'hidden_layer_sizes': hidden_layer_sizes,
        'activation': activation,
        'solver': solver,
        'alpha': alpha,
        'learning_rate': learning_rate
    }
    mlp = MLPClassifier(max_iter=1200, random_state=42)
    search = GridSearchCV(mlp, mlp_params, cv=3, scoring='accuracy', n_jobs=-1)
    search.fit(X_train, y_train)

    best_mlp = search.best_estimator_
    print(f"Best Parameters: {search.best_params_}")
    return best_mlp


def train_svm(X_train, y_train):
    # Define parameter grid with less complex settings
    svm_params = {
        'C': [0.01, 0.1, 1],
        'gamma': ['scale', 0.01, 0.1],
        'kernel': ['linear', 'rbf'],
        'class_weight': ['balanced']
    }
    svm = SVC(max_iter=4000, random_state=42)
    search = GridSearchCV(svm, svm_params, cv=5, scoring='accuracy', n_jobs=-1)
    search.fit(X_train, y_train)

    best_svm = search.best_estimator_
    print(f"Best Parameters: {search.best_params_}")
    return best_svm

def train_gb(X_train, y_train):
    gb_params = {
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5],
        'subsample': [0.8, 1.0]
    }
    gb = GradientBoostingClassifier(random_state=42)
    search = GridSearchCV(gb, gb_params, cv=3, scoring='accuracy', n_jobs=-1)
    search.fit(X_train, y_train)

    best_gb = search.best_estimator_
    print(f"Best Parameters: {search.best_params_}")
    return best_gb

def train_rf(X_train, y_train):
    rf_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced', 'balanced_subsample']
    }
    rf = RandomForestClassifier(random_state=42)
    search = GridSearchCV(rf, rf_params, cv=3, scoring='accuracy', n_jobs=-1)
    search.fit(X_train, y_train)

    best_rf = search.best_estimator_
    print(f"Best Parameters: {search.best_params_}")
    return best_rf

def train_enhanced_nn(X_train, y_train, X_test, y_test):
    print("this model is intended for user to test how hyperparameters affect the accuracy of the model, and how it can be arranged.")
    while True:
        try:
            neurons_layer_1 = int(input("Enter number of neurons in the first hidden layer (e.g., 16):").strip())
            neurons_layer_2 = int(input("Enter number of neurons in the second hidden layer (e.g., 8):").strip())
            dropout_rate = float(input("Enter dropout rate (e.g., 0.4): ").strip())
            l2_reg = float(input("Enter L2 regularization value (e.g., 0.01): ").strip())
            learning_rate_factor = float(input("Enter learning rate reduction factor (e.g., 0.1): ").strip())
            patience_val = int(input("Enter early stopping patience (e.g., 2): ").strip())
            epochs = int(input("Enter number of epochs (e.g., 130): ").strip())
            batch_size = int(input("Enter batch size (e.g., 12): ").strip())
            break  # Break the loop if all inputs are successfully parsed
        except ValueError:
            print("Invalid input. Please enter the correct data type for each parameter.")

    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(neurons_layer_1, activation='relu', kernel_regularizer=l2(l2_reg)),
        Dropout(dropout_rate),
        Dense(neurons_layer_2, activation='relu', kernel_regularizer=l2(l2_reg)),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Learning Rate Reduction and EarlyStopping
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=learning_rate_factor, patience=patience_val, min_lr=1e-6)
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience_val, restore_best_weights=True)

    # Train the model with validation data
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[reduce_lr, early_stopping])

    return model

