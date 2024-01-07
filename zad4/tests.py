from utils import generate_3d_matrix_mesh
from compress import compress_matrix

k = 4
matrix = generate_3d_matrix_mesh(k)
node = compress_matrix(matrix, 5, 0.01)
node.draw_matrix("./loik.png")

   