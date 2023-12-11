from utils import generate_sparse_matrix, get_square_diff, plot_graph
from scipy.sparse.linalg import svds
from compress import compress_matrix, decompress_matrix
import time

SPARSITY_VALUES = [99, 98, 95, 90, 80]
K = 10
TEST_SHAPE = (2**K, 2**K)

TEST_MAX_RANK = [1,1,1,4,4,4]
TEST_MIN_SINGULAR_VAL_IDX = [2, 2**K-2, 2**K/2, 2, 2**K-2, 2**K/2] 

def tests():
    compress_times = []
    decompress_times = []
    diffs = []
    labels = []

    for i, sparsity in enumerate(SPARSITY_VALUES):
        matrix = generate_sparse_matrix(TEST_SHAPE, sparsity)

        U, Sigma, VT = svds(matrix, 2**K - 1)

        for max_rank, min_singular_val_idx in zip(TEST_MAX_RANK, TEST_MIN_SINGULAR_VAL_IDX):
            min_singular_val = Sigma[round(min_singular_val_idx)]
            
            start = time.time()
            compressed_matrix = compress_matrix(matrix, max_rank, min_singular_val)

            end_compress = time.time()
            compress_time = end_compress - start

            new_matrix = decompress_matrix(compressed_matrix)

            end_decompress = time.time()
            decompress_time = end_decompress - end_compress

            compress_times.append(compress_time)
            decompress_times.append(decompress_time)

            label =  "r=" + str(max_rank) + "_ie=" + str(round(min_singular_val_idx))
            labels.append(label)
            img_path = "./img/" + str(sparsity) + "_" + label + '.png' 
            compressed_matrix.draw_matrix(
                img_path,
                "sparsity=" + str(sparsity) + " r=" + str(max_rank) + " e_index=" + str(round(min_singular_val_idx))+ " e=" + str(round(min_singular_val, 4))
            )
            print("done", img_path)

            diffs.append(get_square_diff(matrix, new_matrix))

        l = len(TEST_MAX_RANK)
        plot_graph(labels[i * l: (i+1)* l], compress_times[i * l: (i+1)* l], "Compression Time For Sparsity=" + str(sparsity), "./img/compression_time_sparsity=" + str(sparsity) + ".png")
        plot_graph(labels[i * l: (i+1)* l], decompress_times[i * l: (i+1)* l], "Decompression Time For Sparsity=" + str(sparsity), "./img/decompression_time_sparsity=" + str(sparsity) + ".png")
        plot_graph(labels[i * l: (i+1)* l], diffs[i * l: (i+1)* l], "Compression <-> Decompression Cycle Difference For Sparsity=" + str(sparsity), "./img/cycle_diff_sparsity=" + str(sparsity) + ".png")



# matrix = generate_sparse_matrix((1000, 1000), 99)
# node = compress_matrix(matrix, 5, 0.01)
# node.draw_matrix("./loik.png")
# new_matrix = decompress_matrix(node)
# print(get_square_diff(matrix, new_matrix))


tests()
