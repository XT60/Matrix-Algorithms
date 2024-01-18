import numpy as np
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class DecompositionNode:
    def __init__(self, rows, cols, rank):
        self.rows = rows
        self.cols = cols
        self.rank = rank
        self.U = None
        self.Sigma = None
        self.VT = None
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



def split_matrix(M: np.ndarray):
    rows = M.shape[0] // 2
    return M[:rows, :rows], M[:rows, rows:], M[rows:, :rows], M[rows:, rows:]


def compress_matrix(M, max_rank, min_singular_val):
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


