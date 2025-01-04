# Text Dataset Metaheuristics Pipeline

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from deap import base, creator, tools, algorithms
from mealpy.swarm_based.PSO import OriginalPSO
import tensorflow as tf
import random
import numpy as np
import pickle
import gc

# Ensure TensorFlow uses GPU efficiently
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Load the dataset
dataset_path = 'final_hate_speech.xlsx'
data = pd.read_excel(dataset_path)

# Prepare data
X = data['tweet']
y = data['etiket']

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# TF-IDF transformation
vectorizer = TfidfVectorizer(max_features=2000)  # Adjust as needed
X = vectorizer.fit_transform(X).toarray()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ANN model builder function
def build_ann_model(learning_rate, num_hidden_layers, units_per_layer, activation, dropout_rate, optimizer_name):
    model = Sequential()
    # Input layer
    model.add(Dense(units_per_layer, activation=activation, input_shape=(X_train.shape[1],)))

    # Hidden layers
    for _ in range(num_hidden_layers - 1):
        model.add(Dense(units_per_layer, activation=activation))
        model.add(Dropout(dropout_rate))

    # Output layer
    model.add(Dense(1, activation="sigmoid"))  # Binary classification

    # Optimizer selection
    optimizer = {
        "SGD": tf.keras.optimizers.SGD(learning_rate=learning_rate),
        "Adam": tf.keras.optimizers.Adam(learning_rate=learning_rate),
        "RMSProp": tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
    }[optimizer_name]

    # Compile the model
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    return model

# Evaluate function for GA
def evaluate_ann_ga(individual):
    learning_rate, batch_size, num_hidden_layers, units_per_layer, dropout_rate, activation_index, optimizer_index = individual

    activation_functions = ['relu', 'tanh', 'sigmoid']
    optimizers = ['SGD', 'Adam', 'RMSProp']

    activation = activation_functions[int(activation_index)]
    optimizer = optimizers[int(optimizer_index)]

    model = build_ann_model(
        learning_rate=float(learning_rate),
        num_hidden_layers=int(num_hidden_layers),
        units_per_layer=int(units_per_layer),
        activation=activation,
        dropout_rate=float(dropout_rate),
        optimizer_name=optimizer
    )

    model.fit(X_train, y_train, batch_size=int(batch_size), epochs=3, verbose=0)

    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    return accuracy,

# Genetic Algorithm (GA) setup
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()

toolbox.register("attr_lr", random.uniform, 0.0001, 0.1)
toolbox.register("attr_batch_size", random.choice, [16, 32, 64, 128, 256])
toolbox.register("attr_hidden_layers", random.randint, 1, 5)
toolbox.register("attr_units", random.randint, 16, 128)
toolbox.register("attr_dropout", random.uniform, 0.1, 0.5)
toolbox.register("attr_activation", random.randint, 0, 2)
toolbox.register("attr_optimizer", random.randint, 0, 2)

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_lr, toolbox.attr_batch_size, toolbox.attr_hidden_layers,
                  toolbox.attr_units, toolbox.attr_dropout, toolbox.attr_activation,
                  toolbox.attr_optimizer))

toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_ann_ga)
toolbox.register("mate", tools.cxOnePoint)  # Updated to one-point crossover
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)  # Mutation probability: 5%
toolbox.register("select", tools.selTournament, tournsize=5)  # Tournament size: 5

# Adjusted GA parameters
population_size = 50
num_generations_1000 = 1000
num_generations_5000 = 5000

# Run GA for 1000 generations
print("Running GA for 1000 generations...")
population = toolbox.population(n=population_size)
result_1000 = algorithms.eaSimple(population, toolbox, cxpb=0.3, mutpb=0.05, ngen=num_generations_1000, verbose=True)

with open('ga_ann_progress_1000.pkl', 'wb') as f:
    pickle.dump(population, f)

best_individual_1000 = tools.selBest(population, k=1)[0]
print("Best Individual (GA, 1000 generations):", best_individual_1000)
print("Accuracy (GA, 1000 generations):", evaluate_ann_ga(best_individual_1000))

# Run GA for 5000 generations
print("Running GA for 5000 generations...")
population = toolbox.population(n=population_size)
result_5000 = algorithms.eaSimple(population, toolbox, cxpb=0.3, mutpb=0.05, ngen=num_generations_5000, verbose=True)

with open('ga_ann_progress_5000.pkl', 'wb') as f:
    pickle.dump(population, f)

best_individual_5000 = tools.selBest(population, k=1)[0]
print("Best Individual (GA, 5000 generations):", best_individual_5000)
print("Accuracy (GA, 5000 generations):", evaluate_ann_ga(best_individual_5000))

# PSO Implementation
def fitness_function(solution):
    return evaluate_ann_ga(solution)[0]

problem = {
    'fit_func': fitness_function,
    'lb': [0.0001, 16, 1, 16, 0.1, 0, 0],
    'ub': [0.1, 256, 5, 128, 0.5, 2, 2],
    'minmax': 'max',
    'verbose': True,
}

# Run PSO for 1000 generations
print("Running PSO for 1000 generations...")
model = OriginalPSO(problem, epoch=1000, pop_size=population_size)
best_solution_1000, best_fitness_1000 = model.solve()
print("Best Individual (PSO, 1000 generations):", best_solution_1000)
print("Accuracy (PSO, 1000 generations):", best_fitness_1000)

# Run PSO for 5000 generations
print("Running PSO for 5000 generations...")
model = OriginalPSO(problem, epoch=5000, pop_size=population_size)
best_solution_5000, best_fitness_5000 = model.solve()
print("Best Individual (PSO, 5000 generations):", best_solution_5000)
print("Accuracy (PSO, 5000 generations):", best_fitness_5000)

# Cleanup memory
gc.collect()
