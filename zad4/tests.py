from utils import generate_3d_matrix_mesh, sparse_draw
from compress import compress_matrix
import numpy as np

k = 2
matrix = generate_3d_matrix_mesh(k)
#node = compress_matrix(matrix, 5, 0.01)
sparse_draw(matrix, "./loik.png")

def matrix_to_vectors(matrix):
    vector = np.array([np.count_nonzero(row) for row in matrix])
    return vector

print(matrix_to_vectors(matrix))
