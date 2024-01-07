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
