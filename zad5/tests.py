from utils import get_square_diff, plot_graph, generate_3d_matrix_mesh
from compress import compress_matrix, decompress_matrix
import time
import numpy as np

K_VALUES = [2, 3, 4]

def tests():
    multiply_times = []
    sq_errors = []
    img_path = "./zad5/img/"

    for k in K_VALUES:
        matrix = generate_3d_matrix_mesh(k)

        compressed_matrix = compress_matrix(matrix)
        
        compressed_matrix.draw_matrix(
                img_path + "matrix_" + str(k) + ".png",
                "Matrix for k = " + str(k)                
            )

        vector = np.random.rand(len(matrix))

        start = time.time()
        vec_multiply = compressed_matrix.multiply_by_vector(vector)
        multiply_times.append(time.time() - start)
        print("Multiplied for ", k)

        decompressed_matrix = decompress_matrix(compressed_matrix)
        sq_errors.append(get_square_diff(matrix, decompressed_matrix))

    plot_graph(["K = 2", " K = 3", "K = 4"], multiply_times, "Multiplication Times [ms]" , "./img/multiplication_times.png")
    plot_graph(["K = 2", " K = 3", "K = 4"], sq_errors, "Square Errors" , "./img/square_errors.png")



# matrix = generate_sparse_matrix((1000, 1000), 99)
# node = compress_matrix(matrix, 5, 0.01)
# node.draw_matrix("./loik.png")
# new_matrix = decompress_matrix(node)
# print(get_square_diff(matrix, new_matrix))


tests()
