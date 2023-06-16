# Eviatar Zweigenber, 206299463

import matplotlib.pyplot as plt
import time
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
import numpy as np

def int_to_one_hot(target_num: int, num_classes=10):
    one_hot = np.zeros(num_classes)
    one_hot[target_num] = 1
    return one_hot

def prepare_data(data, target, num_classes=10):
    random_state = check_random_state(1)
    permutation = random_state.permutation(data.shape[0])

    data = data[permutation]
    data.reshape((data.shape[0], -1))
    data = np.hstack((data, np.ones((data.shape[0], 1))))
    
    target = target[permutation]
    # transform target to one-hot encoding
    T = np.zeros((target.shape[0], num_classes))
    inds = np.arange(target.shape[0])
    T[inds, target.astype(int)] = 1
    target = T
    
    return data, target

def split_data(data, target, test_size=0.4):
    X_train, X_test_and_validation, t_train, t_test_and_validation = train_test_split(data, target, test_size=test_size)
    X_validation, X_test, t_validation, t_test = train_test_split(X_test_and_validation, t_test_and_validation, test_size=0.5)
    return X_train, X_validation, X_test, t_train, t_validation, t_test

def standardize_data(data, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
    else:
        data = scaler.transform(data)

    data[:, -1] = 1
    return data, scaler

def randomize_weight_vector(size=(10, 785)):
    # W = np.random.normal(0.5, 1, size=size)
    W = np.random.uniform(0, 1, size=size)
    # W[:, -1] = 1
    np.clip(W, 0, 1, out=W)
    return W

def calc_Y_matrix_form(W, X):
    Z = np.dot(W, X.T)
    Z -= np.max(Z, axis=0) 
    normalizer = np.sum(np.exp(Z), axis=0)
    normalizer = np.vstack(normalizer)
    Y = (1/normalizer)*np.exp(Z).T
    return Y

def cross_entropy_loss(Y, t):
    eps = 1e-10
    return -1*np.sum(t*np.log(Y+eps))

def cross_entropy_loss_for_current_weights(W, X, t):
    Y = calc_Y_matrix_form(W, X)
    return cross_entropy_loss(Y, t)

def cross_entropy_loss_gradient(Y, t, X):
    D = Y - t
    return D.T @ X

def calculate_predictions(W, X):
    Y = calc_Y_matrix_form(W, X)
    return np.argmax(Y, axis=1)

def predict_and_calc_accuracy(W, X, t):
    Y = calc_Y_matrix_form(W, X)
    predictions = np.argmax(Y, axis=1)
    ground_truths = np.argmax(t, axis=1)
    accuracy = np.sum(predictions == ground_truths)/X.shape[0]
    return accuracy

# load dataset
print("Loading dataset...")

start_time = time.time()
mnist = fetch_openml('mnist_784')

print(f"Dataset loaded in {time.time() - start_time:.2f} seconds\n")

# prepare samples and labels
X = mnist['data'].astype('float64').to_numpy()
t = mnist['target'].astype('int').to_numpy()
X, t = prepare_data(X, t)

vec_length = X.shape[1]
num_categories = t.shape[1]

# split data to train, validation and test sets
X_train, X_validation, X_test, t_train, t_validation, t_test = split_data(X, t)

# The next lines standardize the images
X_train, scaler = standardize_data(X_train)
X_test, _ = standardize_data(X_test, scaler)
X_validation, _ = standardize_data(X_validation, scaler)

eta = 0.8
dA_min = 0.001
prev_accuracy = -np.inf
no_improvement_counter = 0

training_set_loss_history = []
validation_set_accuracy_history = []
learning_rate_history = []

# minimize loss by gradient descent
start_time = time.time()
W_cur = randomize_weight_vector(size=(num_categories, vec_length))
iteration_number = 0
while True:
    Y_cur = calc_Y_matrix_form(W_cur, X_train)
    E = cross_entropy_loss(Y_cur, t_train)
    
    accuracy = predict_and_calc_accuracy(W_cur, X_validation, t_validation)
    
    training_set_loss_history.append(E)
    validation_set_accuracy_history.append(accuracy*100)
    
    print(f"Iteration #{iteration_number}:\n\tTraining Loss: {E:.2f}\n" + \
          f"\tValidation Accuracy: {accuracy:.5f}\n" + \
          f"\tdA: {accuracy - prev_accuracy:.5f}\n" + \
          f"\tCurrent Learning Rate: {eta:.10f}\n")
          
    # learning rate is too high, causes divergence when closing in on local minimum
    if accuracy < prev_accuracy:
        eta = 0.5*eta
    elif abs(accuracy - prev_accuracy) < dA_min:
        no_improvement_counter += 1

    learning_rate_history.append(eta)

    if no_improvement_counter > 5:
        break

    prev_accuracy = accuracy
    iteration_number += 1

    # calc gradient and update W
    G = cross_entropy_loss_gradient(Y_cur, t_train, X_train)
    W_cur = W_cur - eta*G

W = W_cur
print(f"Training completed in: {time.time() - start_time:.2f} seconds\n")

print(f"Training Set Accuracy: {predict_and_calc_accuracy(W, X_train, t_train)*100:.2f}%")
print(f"Validation Set Accuracy: {predict_and_calc_accuracy(W, X_validation, t_validation)*100:.2f}%")
print(f"Test Set Accuracy: {predict_and_calc_accuracy(W, X_test, t_test)*100:.2f}%")

fig, axs = plt.subplots(3, 1, sharex=True, tight_layout=True)
axs = np.ravel(axs)

axs[0].plot(training_set_loss_history)
axs[0].set_title("Training Set Loss")
axs[0].set_ylabel("Loss")

axs[1].plot(validation_set_accuracy_history)
axs[1].set_title(f"Validation Set Accuracy - final accuracy: {validation_set_accuracy_history[-1]:.2f}%")
axs[1].set_ylabel("Accuracy (%)")

axs[2].plot(learning_rate_history)
axs[2].set_title("Learning Rate")
axs[2].set_xlabel("Iteration #")
axs[2].set_ylabel("$\eta$")
plt.show()
