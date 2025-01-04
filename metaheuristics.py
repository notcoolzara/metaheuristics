import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from deap import base, creator, tools, algorithms
from mealpy.swarm_based.PSO import OriginalPSO
import tensorflow as tf
import random
import numpy as np
import pickle

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
vectorizer = TfidfVectorizer(max_features=2000)  # Reduced to save memory
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
        model.add(Dropout(dropout_rate))  # Apply dropout to avoid overfitting

    # Output layer
    model.add(Dense(1, activation="sigmoid"))  # For binary classification

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
    # Extract hyperparameters from the individual
    learning_rate, batch_size, num_hidden_layers, units_per_layer, dropout_rate, activation_index, optimizer_index = individual

    # Define activation functions and optimizers
    activation_functions = ['relu', 'tanh', 'sigmoid']
    optimizers = ['SGD', 'Adam', 'RMSProp']

    # Select activation function and optimizer based on the individual's indexes
    activation = activation_functions[int(activation_index)]
    optimizer = optimizers[int(optimizer_index)]

    # Build the ANN model with the current hyperparameters
    model = build_ann_model(
        learning_rate=float(learning_rate),
        num_hidden_layers=int(num_hidden_layers),
        units_per_layer=int(units_per_layer),
        activation=activation,
        dropout_rate=float(dropout_rate),
        optimizer_name=optimizer
    )

    # Train the model (set verbose=0 to suppress output)
    model.fit(X_train, y_train, batch_size=int(batch_size), epochs=3, verbose=0)

    # Evaluate the model on the test set
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)

    return accuracy,  # Return as a tuple (required for DEAP)

# GA setup
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()

# Register hyperparameter attributes
toolbox.register("attr_lr", random.uniform, 0.0001, 0.1)
toolbox.register("attr_batch_size", random.choice, [16, 32, 64, 128, 256])
toolbox.register("attr_hidden_layers", random.randint, 1, 5)
toolbox.register("attr_units", random.randint, 16, 128)
toolbox.register("attr_dropout", random.uniform, 0.1, 0.5)
toolbox.register("attr_activation", random.randint, 0, 2)
toolbox.register("attr_optimizer", random.randint, 0, 2)

# Define an individual and population
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_lr, toolbox.attr_batch_size, toolbox.attr_hidden_layers,
                  toolbox.attr_units, toolbox.attr_dropout, toolbox.attr_activation,
                  toolbox.attr_optimizer))

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate_ann_ga)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Run GA
population = toolbox.population(n=20)  # Reduced population size
result = algorithms.eaSimple(population, toolbox, cxpb=0.3, mutpb=0.05, ngen=20, verbose=True)

# Save progress
with open('ga_ann_progress.pkl', 'wb') as f:
    pickle.dump(population, f)

# Get the best individual
best_individual = tools.selBest(population, k=1)[0]
print("Best Individual (GA):", best_individual)
print("Accuracy (GA):", evaluate_ann_ga(best_individual))

# PSO implementation
def fitness_function(solution):
    return evaluate_ann_ga(solution)[0]

# Define PSO parameters
problem = {
    'fit_func': fitness_function,
    'lb': [0.0001, 16, 1, 16, 0.1, 0, 0],  # Lower bounds
    'ub': [0.1, 256, 5, 128, 0.5, 2, 2],  # Upper bounds
    'minmax': 'max',
    'verbose': True,
}

model = OriginalPSO(problem, epoch=20, pop_size=20)
best_solution, best_fitness = model.solve()
print("Best Individual (PSO):", best_solution)
print("Accuracy (PSO):", best_fitness)
