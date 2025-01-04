# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, AveragePooling2D, Flatten
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

# CNN model builder function
def build_cnn_model(learning_rate, num_conv_layers, filters_per_layer, kernel_size, activation, pooling_type, optimizer_name):
    model = Sequential()

    # Add convolutional and pooling layers
    for _ in range(num_conv_layers):
        model.add(Conv2D(filters=filters_per_layer, kernel_size=kernel_size, activation=activation, input_shape=(X_train.shape[1], 1, 1)))
        if pooling_type == 'MaxPooling':
            model.add(MaxPooling2D(pool_size=(2, 2)))
        else:
            model.add(AveragePooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    # Fully connected layer
    model.add(Dense(64, activation=activation))
    model.add(Dropout(0.5))  # Dropout to prevent overfitting

    # Output layer
    model.add(Dense(1, activation="sigmoid"))

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
def evaluate_cnn_ga(individual):
    learning_rate, batch_size, num_conv_layers, filters_per_layer, kernel_size_index, activation_index, pooling_index, optimizer_index = individual

    # Define hyperparameter options
    kernel_sizes = [(3, 3), (5, 5)]
    activation_functions = ['relu', 'tanh', 'sigmoid']
    pooling_types = ['MaxPooling', 'AveragePooling']
    optimizers = ['SGD', 'Adam', 'RMSProp']

    # Map indices to hyperparameter values
    kernel_size = kernel_sizes[int(kernel_size_index)]
    activation = activation_functions[int(activation_index)]
    pooling_type = pooling_types[int(pooling_index)]
    optimizer = optimizers[int(optimizer_index)]

    # Build and train the CNN model
    model = build_cnn_model(
        learning_rate=float(learning_rate),
        num_conv_layers=int(num_conv_layers),
        filters_per_layer=int(filters_per_layer),
        kernel_size=kernel_size,
        activation=activation,
        pooling_type=pooling_type,
        optimizer_name=optimizer
    )

    model.fit(X_train, y_train, batch_size=int(batch_size), epochs=3, verbose=0)
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    return accuracy,  # Return as a tuple for DEAP

# GA setup
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()

# Register hyperparameter attributes
toolbox.register("attr_lr", random.uniform, 0.0001, 0.1)
toolbox.register("attr_batch_size", random.choice, [16, 32, 64, 128, 256])
toolbox.register("attr_conv_layers", random.randint, 1, 5)
toolbox.register("attr_filters", random.randint, 8, 128)
toolbox.register("attr_kernel_size", random.randint, 0, 1)
toolbox.register("attr_activation", random.randint, 0, 2)
toolbox.register("attr_pooling", random.randint, 0, 1)
toolbox.register("attr_optimizer", random.randint, 0, 2)

# Define an individual and population
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_lr, toolbox.attr_batch_size, toolbox.attr_conv_layers,
                  toolbox.attr_filters, toolbox.attr_kernel_size, toolbox.attr_activation,
                  toolbox.attr_pooling, toolbox.attr_optimizer))

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate_cnn_ga)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Run GA
population = toolbox.population(n=20)  # Reduced population size
result = algorithms.eaSimple(population, toolbox, cxpb=0.3, mutpb=0.05, ngen=20, verbose=True)

# Save progress
with open('ga_cnn_progress.pkl', 'wb') as f:
    pickle.dump(population, f)

# Get the best individual
best_individual = tools.selBest(population, k=1)[0]
print("Best Individual (GA):", best_individual)
print("Accuracy (GA):", evaluate_cnn_ga(best_individual))

# PSO implementation
def fitness_function(solution):
    return evaluate_cnn_ga(solution)[0]

# Define PSO parameters
problem = {
    'fit_func': fitness_function,
    'lb': [0.0001, 16, 1, 8, 0, 0, 0, 0],  # Lower bounds
    'ub': [0.1, 256, 5, 128, 1, 2, 1, 2],  # Upper bounds
    'minmax': 'max',
    'verbose': True,
}

model = OriginalPSO(problem, epoch=20, pop_size=20)
best_solution, best_fitness = model.solve()
print("Best Individual (PSO):", best_solution)
print("Accuracy (PSO):", best_fitness)
