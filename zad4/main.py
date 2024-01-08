from compress import compress_matrix, decompress_matrix
from utils import generate_3d_matrix_mesh, sparse_draw
from permutation_engine import PermutationEngine

for k in [2, 3, 4]:
    matrix = generate_3d_matrix_mesh(k)

    #draw matrix before compression
    sparse_draw(matrix, f"./zad4/data/k_{k}/2a.png",
                title=f'k = {k} Matrix')

    #draw matrix after compression
    compressed_matrix = compress_matrix(matrix.copy(), max_rank=4, min_singular_val=0)
    compressed_matrix.draw_matrix(f"./zad4/data/k_{k}/2b.png",
                                  title=f'k = {k} Matrix after compression')
    

    # Minimum-Degree
    permutation_engine = PermutationEngine('minimum_degree')

    permutated_matrix = permutation_engine.permutate(matrix.copy())
    sparse_draw(permutated_matrix, f"./zad4/data/k_{k}/2c-min_deg.png",
                title=f'k = {k} Matrix after Minimum-Degree Algorithm')
    

    #draw matrix after compressiona and permutation
    compressed_permutated_matrix = compress_matrix(permutated_matrix, max_rank=4, min_singular_val=0)
    compressed_permutated_matrix.draw_matrix(f"./zad4/data/k_{k}/2d-min_deg.png",
                                             title=f'k = {k} Matrix after Minimum-Degree Algorithm and compression')



    # Reversed-Cuthil-Mckee
    #draw matrix after permutation
    permutation_engine = PermutationEngine('reversed_cuthill_mckee')

    permutated_matrix = permutation_engine.permutate(matrix.copy())
    sparse_draw(permutated_matrix, f"./zad4/data/k_{k}/2c-rev_cut_mc.png",
                title=f'k = {k} Matrix after reversed Cuthill & McKee Algorithm')
    

    #draw matrix after compressiona and permutation
    compressed_permutated_matrix = compress_matrix(permutated_matrix, max_rank=4, min_singular_val=0)
    compressed_permutated_matrix.draw_matrix(f"./zad4/data/k_{k}/2d-rev_cut_mc.png", 
                                             title=f'k = {k} Matrix after reversed Cuthill & McKee Algorithm and compression')


    # Cuthil-Mckee
    #draw matrix after permutation
    permutation_engine = PermutationEngine('cuthill_mckee')

    permutated_matrix = permutation_engine.permutate(matrix.copy())
    sparse_draw(permutated_matrix, f"./zad4/data/k_{k}/2c-cut_mc.png",
                title=f'k = {k} Matrix after Cuthill & McKee Algorithm')
    

    #draw matrix after compressiona and permutation
    compressed_permutated_matrix = compress_matrix(permutated_matrix, max_rank=4, min_singular_val=0)
    compressed_permutated_matrix.draw_matrix(f"./zad4/data/k_{k}/2d-cut_mc.png",
                                             title=f'k = {k} Matrix after Cuthill & McKee Algorithm and compression')
