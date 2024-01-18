import numpy as np
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class DecompositionNode:
    def __init__(self, rows, cols, rank):
        self.rows = rows
        self.cols = cols
        self.rank = rank
        self.U = None # columns
        self.Sigma = None
        self.VT = None # rows
        self.full_matrix = None
        self.children = []

    def __get_matrix_to_draw(self):
        if self.rank is not None:
            if self.rank > 0:
                matrix = np.zeros((self.rows, self.cols))
                matrix[:, :self.rank] = 1
                matrix[:self.rank, :] = 1
                return matrix
            else:
                return np.zeros((self.rows, self.cols))
        elif self.full_matrix is not None:
            return np.ones((self.rows, self.cols))
        else:
            return np.vstack(
                (
                    np.hstack((self.children[0].__get_matrix_to_draw(), self.children[1].__get_matrix_to_draw())),
                    np.hstack((self.children[2].__get_matrix_to_draw(), self.children[3].__get_matrix_to_draw())),
                )
            )

    def draw_matrix(self, img_name=None, title = "Bar plot"):
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.matshow(self.__get_matrix_to_draw(), cmap=ListedColormap(['w', 'k']))
        if img_name:
            ax.set_title(title)  # Set the title using ax.set_title
            plt.savefig(img_name)
            plt.close()

    
    def multiply_by_vector(self, vector):
        # check if the vector is of valid format
        vector = np.squeeze(vector)
        n = vector.shape[0]
        if len(vector.shape) > 2:
            raise ValueError("Vector should be one dimensional")
        if n != self.rows:
            raise ValueError(f"Vector length should be {self.length}")

        # directly multiply
        if len(self.children) == 0:
            if self.full_matrix is not None:
                return np.squeeze(self.full_matrix @ vector.reshape(-1, 1))
            else:
                U = self.U if self.U is not None else np.zeros(self.rows, dtype=float) 
                VT = self.VT if self.VT is not None else np.zeros(self.cols, dtype=float)
                return np.squeeze(U * np.dot(VT, vector))
        
        # divide vector and call multiplication recursively
        else:
            half_length = (len(vector)//2)
            upper_input_vector = vector[:half_length]
            lower_input_vector = vector[half_length:]

            upper_output_vector = np.squeeze(self.children[0].multiply_by_vector(upper_input_vector) + self.children[1].multiply_by_vector(upper_input_vector))
            lower_output_vector = np.squeeze(self.children[2].multiply_by_vector(lower_input_vector) + self.children[3].multiply_by_vector(lower_input_vector))

            return np.concatenate((upper_output_vector, lower_output_vector), axis=0)


def split_matrix(M: np.ndarray):
    rows = M.shape[0] // 2
    return M[:rows, :rows], M[:rows, rows:], M[rows:, :rows], M[rows:, rows:]


def compress_matrix(M, max_rank = 1, min_singular_val = 1):
    if not np.any(M):
        return DecompositionNode(*M.shape, 0)

    if min(M.shape[0], M.shape[1]) <= max_rank + 1:
        # Matrix is small enough, no need for further compression
        node = DecompositionNode(*M.shape, None)
        node.full_matrix = M
        return node

    U, Sigma, VT = svds(M, max_rank + 1)

    if abs(Sigma[0]) < min_singular_val:
        # Compress
        node = DecompositionNode(*M.shape, max_rank)
        node.U = U[:, 1:]
        node.Sigma = Sigma[1:]
        node.Sigma[node.Sigma < min_singular_val] = 0
        node.VT = VT[1:]
    else:
        # Divide
        M11, M12, M21, M22 = split_matrix(M)
        node = DecompositionNode(*M.shape, None)
        node.children = [
            compress_matrix(M11, max_rank, min_singular_val),
            compress_matrix(M12, max_rank, min_singular_val),
            compress_matrix(M21, max_rank, min_singular_val),
            compress_matrix(M22, max_rank, min_singular_val)
        ]

    return node


def decompress_matrix(node):
    if node.rank is not None:
        if node.rank > 0:
            return node.U @ (np.diag(node.Sigma) @ node.VT)
        else:
            return np.zeros((node.rows, node.cols))
    elif node.full_matrix is not None:
        return node.full_matrix
    else:
        return np.vstack(
            (
                np.hstack((decompress_matrix(node.children[0]), decompress_matrix(node.children[1]))),
                np.hstack((decompress_matrix(node.children[2]), decompress_matrix(node.children[3]))),
            )
        )



# matrix = np.array([[1.0, 2.0, 3.0, 4.0],
#                    [5.0, 6.0, 7.0, 8.0],
#                    [9.0, 10.0, 11.0, 12.0],
#                    [13.0, 14.0, 15.0, 16.0]])
# vector = np.array([1.0, 1.0, 1.0, 1.0])

# M = compress_matrix(matrix)
# res = M.multiply_by_vector(vector)
# print(res)
# print(matrix @ vector)
