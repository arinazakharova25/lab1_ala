import matplotlib.pyplot as plt
import numpy as np

lynx = np.array([
[209.70, 368.42], [157.63, 332.16], [118.82, 284.21], [80.95, 224.56], [43.08, 244.44], [20.36, 266.67], [-4.26, 293.57], [2.37, 263.16], [-20.36, 292.40], [-39.29, 299.42], [-21.30, 259.65],
[-50.65, 267.84], [-39.29, 242.11], [-55.38, 240.94], [-100.83, 300.58], [-149.11, 345.03], [-172.78, 361.40], [-189.82, 300.58], [-192.66, 225.73], [-181.30, 145.03], [-168.05, 104.09], [-184.14, 66.67],
[-186.98, 31.58], [-183.20, 3.51], [-208.76, -4.68], [-197.40, -29.24], [-182.25, -44.44], [-203.08, -43.27], [-172.78, -92.40], [-131.12, -126.32], [-101.78, -147.37], [-74.32, -163.74], [-110.30, -224.56],
[-143.43, -287.72], [-161.42, -240.94], [-282.60, -221.05], [-388.64, -205.85], [-370.65, -301.75], [-339.41, -397.66], [18.46, -397.66], [345.09, -400.00], [359.29, -378.95], [367.81, -342.69], [346.98, -362.57],
[363.08, -302.92], [357.40, -243.27], [348.88, -266.67], [336.57, -201.17], [290.18, -135.67], [240.00, -118.13], [258.93, -164.91], [257.99, -228.07], [252.31, -271.35], [256.09, -333.33],
[247.57, -359.06], [230.53, -307.60], [194.56, -238.60], [160.47, -181.29], [120.71, -149.71], [165.21, -132.16], [201.18, -100.58], [183.20, -99.42], [221.07, -73.68], [253.25, -24.56], [222.01, -23.39],
[251.36, -1.17], [262.72, 24.56], [234.32, 25.73], [214.44, 42.11], [202.13, 60.82], [220.12, 101.75], [234.32, 160.23], [240.00, 230.41], [232.43, 316.96]
])

X = lynx.T
print(lynx.shape, X.shape)


def stretch(X, a, b):
    X = X.copy()
    A = np.array([[a, 0],
                  [0, b]])
    return A @ X, A


def shear(X, a, b):
    X = X.copy()
    A = np.array([[1, a],
                  [b, 1]])
    return A @ X, A

def reflection(X, a, b):
    X = X.copy()
    A = 1/(a**2 + b**2) * np.array([[a**2 - b**2, 2*a*b],
                                    [2*a*b, b**2 - a**2]])
    return A @ X, A


def rotation(X, theta):
    X = X.copy()
    A = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    return A @ X, A

def show(results, title=""):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    for ax, (name, Y, A) in zip(np.atleast_1d(axes), results):
        ax.fill(X[0], X[1], color="gray", alpha=0.3)
        ax.fill(Y[0], Y[1], color="blue", alpha=0.4)
        ax.axhline(0, color="k", lw=0.5)
        ax.axvline(0, color="k", lw=0.5)
        ax.set_aspect("equal")
        ax.set_title(name, fontsize=10)
        print(f"{name}\n{np.round(A, 3)}\n")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    tests = {
        "Stretch": (stretch, [(1.5, 0.7), (2, 2), (-1, 1), (0, 1)]),
        "Shear": (shear, [(0.5, 0), (0, 0.5), (0.5, 0.5), (1, 1)]),
        "Reflection": (reflection, [(1, 1), (1, 0), (0, 1), (1, -1)]),
        "Rotation": (rotation, [(np.pi / 4,), (-np.pi / 4,), (np.pi / 2,), (np.pi,)]),
    }
    for name, (func, params) in tests.items():
        results = []
        for p in params:
            Y, A = func(X, *p)
            results.append((f"{name} {tuple(np.round(p, 2))}", Y, A))
        show(results, name)

    _, S = stretch(X, 1.5, 0.7)
    _, H = shear(X, 0.5, 0)
    _, R = rotation(X, np.pi / 4)

    A1 = R @ H @ S
    A2 = S @ H @ R
    A3 = R @ S @ H

    show([("Order 1: R @ H @ S (S → H → R)", A1 @ X, A1),
          ("Order 2: S @ H @ R (R → H → S)", A2 @ X, A2),
          ("Order 3: R @ S @ H (H → S → R)", A3 @ X, A3)],
         "Task 2: порядок перетворень")

    print("A1 == A2", np.allclose(A1, A2))
    print("A1 == A3", np.allclose(A1, A3))
    print("A2 == A3", np.allclose(A2, A3))
    print("det:", round(np.linalg.det(A1), 3), round(np.linalg.det(A2), 3), round(np.linalg.det(A3), 3))