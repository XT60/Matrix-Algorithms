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
    if len(labels) < len(values):
        raise ValueError("Number of labels must match the number of values.")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(labels[: len(values)], values, color='blue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if file_name == None:
        plt.show()
    else:
        plt.savefig(file_name)
        plt.close()

import numpy as np
import matplotlib.pyplot as plt

def generate_3d_matrix_mesh(k):
    size = 2**k
    v_num = 2**(3*k)
    
    mesh = np.zeros((v_num, v_num))

    dx = 1
    dy = 2**k
    dz = 2**(2*k)

    for z in range(size):
        for y in range(size):
            for x in range(size):
                idx = x + y*size + z*size*size
                # x_plane
                _complete_connections_in_plane(position=x, step=dx, index=idx, mesh=mesh, size=size) 
                # y_plane
                _complete_connections_in_plane(position=y, step=dy, index=idx, mesh=mesh, size=size) 
                # z_plane
                _complete_connections_in_plane(position=z, step=dz, index=idx, mesh=mesh, size=size)
                
    return mesh



def _complete_connections_in_plane(position, step, index, mesh, size):
    if position != 0:
        mesh[index][index-step] = np.random.random()
    if position != size-1:
        mesh[index][index+step] = np.random.random()


# def sparse_draw(matrix, img_name, title = ''):
#     image = (matrix == 0)
#     image = image.astype(int)  *255
#     fig = plt.figure()
#     n = len(matrix)//50
#     fig.set_size_inches((n,n))
#     plt.imshow(image,cmap = "gray", vmin=0, vmax=255)
#     plt.title(title)
#     plt.axis('off')
#     plt.savefig(img_name)

def sparse_draw(matrix, img_name=None, title="Bar plot"):
    rows, cols = matrix.shape

    if matrix.ndim != 2:
        raise ValueError("Input must be a 2D numpy array.")


    fig, ax = plt.subplots(figsize=(16, 9))
    ax.matshow(matrix, cmap=plt.cm.get_cmap("binary", 2))  # "binary" colormap with 2 colors (black and white)

    if img_name:
        ax.set_title(title)
        plt.savefig(img_name)
        plt.close()



if __name__ == '__main__':
# Przykład użycia dla k=2
    k_value = 2
    result_matrix = generate_3d_matrix_mesh(k_value)
    sparse_draw(result_matrix, img_name='./loik.png')
    print(result_matrix)
