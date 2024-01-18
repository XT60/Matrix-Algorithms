import numpy as np
import matplotlib.pyplot as plt

def generate_sparse_matrix(shape, sparsity):
    random_matrix = np.random.rand(*shape)
    threshold = np.percentile(random_matrix, sparsity)
    sparse_matrix = np.where(random_matrix > threshold, random_matrix, 0)
    return sparse_matrix

def get_square_diff(m1, m2):
    squared_diff = np.square(m1 - m2)
    return np.sum(squared_diff)

def plot_graph(labels, values, title="Bar Plot", file_name= None, xlabel="Labels", ylabel="Values"):
    # Check if the length of labels and values match
    if len(labels) != len(values):
        raise ValueError("Number of labels must match the number of values.")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color='blue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if file_name == None:
        plt.show()
    else:
        plt.savefig(file_name)
        plt.close()
