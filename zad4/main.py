from compress import compress_matrix, decompress_matrix
from utils import generate_3d_matrix_mesh, sparse_draw
from permutations import cuthill_mckee, reversed_cuthill_mckee, minimum_degree_permutation, permutate

for k in [2, 3, 4]:
    matrix = generate_3d_matrix_mesh(k)

    #draw matrix before compression
    sparse_draw(matrix, f"./zad4/data/k_{k}/2a.png")

    #draw matrix after compression
    compressed_matrix = compress_matrix(matrix, max_rank=4, min_singular_val=0)
    compressed_matrix.draw_matrix(f"./zad4/data/k_{k}/2b.png")
    

    #draw matrix after permutation
    permutation = cuthill_mckee(matrix)

    permutated_matrix = permutate(matrix, permutation)
    sparse_draw(permutated_matrix, f"./zad4/data/k_{k}/2c.png")
    

    #draw matrix after compressiona and permutation
    compressed_permutated_matrix = compress_matrix(permutated_matrix, max_rank=4, min_singular_val=0)
    compressed_permutated_matrix.draw_matrix(f"./zad4/data/k_{k}/2d.png")

