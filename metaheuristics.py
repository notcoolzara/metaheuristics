# Install necessary libraries (uncomment if needed)
# !pip install deap
# !pip install mealpy
# !pip install --upgrade mealpy

# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from deap import base, creator, tools, algorithms
import random
import numpy as np
from mealpy.swarm_based import BasePSO
import pickle

# Load the dataset
dataset_path = '<path_to_dataset>'
data = pd.read_excel(dataset_path)

# Prepare data
X = data['tweet']
y = data['etiket']

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# TF-IDF transformation
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X).toarray()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ANN model builder function
def build_model(learning_rate, num_hidden_layers, units_per_layer, activation, dropout_rate, optimizer_name):
    model = Sequential()
    model.add(Dense(units_per_layer, activation=activation, input_shape=(X_train.shape[1],)))
    for _ in range(num_hidden_layers - 1):
        model.add(Dense(units_per_layer, activation=activation))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation="sigmoid"))

    optimizer = {
        "SGD": tf.keras.optimizers.SGD(learning_rate=learning_rate),
        "Adam": tf.keras.optimizers.Adam(learning_rate=learning_rate),
        "RMSProp": tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
    }[optimizer_name]

    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return model

# Genetic Algorithm (GA) Evaluation Function
def evaluate_ga(individual):
    learning_rate, batch_size, num_hidden_layers, units_per_layer, dropout_rate, activation_index, optimizer_index = individual
    activation_functions = ['relu', 'tanh', 'sigmoid']
    optimizers = ['SGD', 'Adam', 'RMSProp']

    activation = activation_functions[int(activation_index)]
    optimizer = optimizers[int(optimizer_index)]

    model = build_model(
        learning_rate=float(learning_rate),
        num_hidden_layers=int(num_hidden_layers),
        units_per_layer=int(units_per_layer),
        activation=activation,
        dropout_rate=float(dropout_rate),
        optimizer_name=optimizer
    )
    model.fit(X_train, y_train, batch_size=int(batch_size), epochs=5, verbose=0)
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    return accuracy,

# GA Setup
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()

toolbox.register("attr_lr", random.uniform, 0.0001, 0.1)
toolbox.register("attr_batch_size", random.choice, [16, 32, 64, 128, 256])
toolbox.register("attr_hidden_layers", random.randint, 1, 5)
toolbox.register("attr_units", random.choice, [16, 32, 64, 128])
toolbox.register("attr_dropout", random.uniform, 0.1, 0.5)
toolbox.register("attr_activation", random.randint, 0, 2)
toolbox.register("attr_optimizer", random.randint, 0, 2)

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_lr, toolbox.attr_batch_size, toolbox.attr_hidden_layers,
                  toolbox.attr_units, toolbox.attr_dropout,
                  toolbox.attr_activation, toolbox.attr_optimizer))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_ga)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=1, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=5)

# Run GA
population = toolbox.population(n=50)
result = algorithms.eaSimple(population, toolbox, cxpb=0.3, mutpb=0.05, ngen=1000, verbose=True)

# Save progress
with open('ga_progress.pkl', 'wb') as f:
    pickle.dump(population, f)

# Get the best individual
best_individual = tools.selBest(population, k=1)[0]
print("Best Individual (GA):", best_individual)
print("Accuracy (GA):", evaluate_ga(best_individual))

# PSO Setup
def pso_model(solution):
    learning_rate, num_hidden_layers, num_neurons, activation_index = solution
    activation_functions = ['relu', 'tanh', 'sigmoid']
    activation = activation_functions[int(activation_index)]

    model = build_model(
        learning_rate=learning_rate,
        num_hidden_layers=int(num_hidden_layers),
        units_per_layer=int(num_neurons),
        activation=activation,
        dropout_rate=0.3,  # Default dropout rate for PSO
        optimizer_name="Adam"  # Default optimizer for PSO
    )
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    return 1 - accuracy

# PSO Problem Definition
problem_dict = {
    "fit_func": pso_model,
    "lb": [0.0001, 1, 16, 0],
    "ub": [0.1, 5, 128, 2],
    "minmax": "min",
    "log_to": None,
}

# Run PSO
model = BasePSO(problem_dict, epoch=1000, pop_size=50)
best_position, best_fitness = model.train()
print("Best Individual (PSO):", best_position)
print("Accuracy (PSO):", 1 - best_fitness)
