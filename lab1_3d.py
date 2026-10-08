import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def read_off(filename: str):
    with open(filename, 'r') as f:
        if 'OFF' != f.readline().strip():
            raise ValueError('Not a valid OFF header')
        n_verts, n_faces, _ = map(int, f.readline().strip().split())
        verts = [list(map(float, f.readline().strip().split())) for _ in range(n_verts)]
        faces = [list(map(int, f.readline().strip().split()[1:])) for _ in range(n_faces)]
    return np.array(verts), faces

def plot_off(vertices, faces, title="3D Model"):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection([vertices[face] for face in faces], alpha=0.3, edgecolor='k')
    ax.add_collection3d(mesh)
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=2, c='r')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

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

if __name__ == "__main__":
    file_path = "/Users/arina/.cache/kagglehub/datasets/balraj98/modelnet40-princeton-3d-object-dataset/versions/1/ModelNet40/airplane/test/airplane_0630.off"
    vertices, faces = read_off(file_path)
    print(vertices.shape)

    plot_off(vertices, faces, "Original")

    V, A = rotate_xy(vertices, np.pi / 4)
    print("xy rotation 45°:\n", np.round(A, 3))
    plot_off(V, faces, "Rotation 45° xy (about z)")

    V, A = rotate_yz(vertices, np.pi / 4)
    print("yz rotation 45°:\n", np.round(A, 3))
    plot_off(V, faces, "Rotation 45° yz (about x)")

    V, A = rotate_xz(vertices, np.pi / 4)
    print("xz rotation 45°:\n", np.round(A, 3))
    plot_off(V, faces, "Rotation 45° xz (about y)")