import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

file_path = "/Users/arina/.cache/kagglehub/datasets/balraj98/modelnet40-princeton-3d-object-dataset/versions/1/ModelNet40/airplane/test/airplane_0630.off"

def read_off(filename: str):
    with open(filename, 'r') as f:
        if 'OFF' != f.readline().strip():
            raise ValueError('Not a valid OFF header')
        n_verts, n_faces, _ = map(int, f.readline().strip().split())
        verts = [list(map(float, f.readline().strip().split())) for _ in range(n_verts)]
        faces = [list(map(int, f.readline().strip().split()[1:])) for _ in range(n_faces)]
    return np.array(verts), faces

def plot_off(vertices, faces, title="3D Model", lim=None):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection([vertices[face] for face in faces], alpha=0.3, edgecolor='k')
    ax.add_collection3d(mesh)
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=2, c='r')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    if lim is not None:
        ax.set_xlim([-lim, lim])
        ax.set_ylim([-lim, lim])
        ax.set_zlim([-lim, lim])
    else:
        ax.auto_scale_xyz(vertices[:, 0], vertices[:, 1], vertices[:, 2])

    ax.set_title(title)
    plt.show()

def rotate_xy (X, theta):
    X = X.copy()
    A = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0],
                  [0, 0, 1]])
    return (A @ X.T).T, A


def rotate_yz(X, theta):
    X = X.copy()
    A = np.array([[1, 0, 0],
                  [0, np.cos(theta), -np.sin(theta)],
                  [0, np.sin(theta), np.cos(theta)]])
    return (A @ X.T).T, A


def rotate_xz(X, theta):
    X = X.copy()
    A = np.array([[np.cos(theta), 0, -np.sin(theta)],
                  [0, 1, 0],
                  [np.sin(theta), 0, np.cos(theta)]])
    return (A @ X.T).T, A


def check_rotation(A, name=""):
    det = np.linalg.det(A)
    is_orthogonal = np.allclose(A.T @ A, np.eye(3))
    print(f"[{name}] det(A) = {det:.3f}, Ортогональна (A^T * A = I): {is_orthogonal}")


if __name__ == "__main__":

    vertices, faces = read_off(file_path)
    print("Форма вершин:", vertices.shape)
    print("Центр моделі:", np.round(vertices.mean(axis=0), 3))

    lim = np.linalg.norm(vertices, axis=1).max()

    plot_off(vertices, faces, "Original", lim)

    angle = np.pi / 4

    V, A = rotate_xy(vertices, angle)
    print("Матриця rotate_xy (45°):\n", np.round(A, 3))
    check_rotation(A, "rotate_xy")
    plot_off(V, faces, "Rotation 45° xy (about z)", lim)

    V, A = rotate_yz(vertices, angle)
    print("Матриця rotate_yz (45°):\n", np.round(A, 3))
    check_rotation(A, "rotate_yz")
    plot_off(V, faces, "Rotation 45° yz (about x)", lim)

    V, A = rotate_xz(vertices, angle)
    print("Матриця rotate_xz (45°):\n", np.round(A, 3))
    check_rotation(A, "rotate_xz")
    plot_off(V, faces, "Rotation 45° xz (about y)", lim)

    a_yz, a_xz, a_xy = np.deg2rad(30), np.deg2rad(45), np.deg2rad(60)
    _, A_yz = rotate_yz(vertices, a_yz)
    _, A_xz = rotate_xz(vertices, a_xz)
    _, A_xy = rotate_xy(vertices, a_xy)

    A_total = A_xy @ A_xz @ A_yz
    print("Матриця композиції (xy 60° @ xz 45° @ yz 30°):\n", np.round(A_total, 3))
    check_rotation(A_total, "композиція")

    V_total = (A_total @ vertices.copy().T).T
    plot_off(V_total, faces, "Composition: yz 30° → xz 45° → xy 60°", lim)

    V_seq, _ = rotate_yz(vertices, a_yz)
    V_seq, _ = rotate_xz(V_seq, a_xz)
    V_seq, _ = rotate_xy(V_seq, a_xy)
    print("Матриця композиції == послідовне застосування:", np.allclose(V_total, V_seq))

    A_other = A_yz @ A_xz @ A_xy
    print("Інший порядок (xy → xz → yz) дає ту саму матрицю?", np.allclose(A_total, A_other))