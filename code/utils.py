import numpy as np
from matplotlib import pyplot as plt

def open_file(name, class_description: bool):
    X = []
    Y = []
    with open(name, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            values = list(map(float, parts))
            if class_description:
                x = values[:-1]
                y = values[-1]
                X.append(np.array(x))
                Y.append(int(y))
            else:
                X.append(np.array(values))
    return np.array(X), np.array(Y)

def plot_2d_classes(X, Y, title):

    classes = np.unique(Y)
    colors = plt.cm.get_cmap('tab20_r', len(classes))

    plt.figure(figsize=(8, 6))
    for idx, cls in enumerate(classes):
        plt.scatter(X[Y == cls, 0], X[Y == cls, 1],
                    color=colors(idx), label=f'Class {cls}', s=30, edgecolors='k')

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(title)
    plt.grid(True)
    plt.axis('equal')
    plt.show()


def labeling_accuracy(Y_pred, Y):
    n = len(Y)
    correct = 0
    total = 0

    for i in range(n):
        for j in range(i + 1, n):
            same_true = Y[i] == Y[j]
            same_pred = Y_pred[i] == Y_pred[j]
            if same_true == same_pred:
                correct += 1
            total += 1

    return correct / total if total > 0 else 0.0


def plot_data(X, Y, label, xlabel, ylabel):
    plt.figure(figsize=(6, 5))
    plt.plot(X, Y, marker='o', color='tab:orange')
    plt.title(label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

def corr(data):
    n = data[0].shape[0]
    corr_matrix = np.corrcoef([data[i][:n] for i in range(len(data))], rowvar=False)
    plt.figure(figsize=(8, 6))
    im = plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(im)

    labels = [f"X{i}" for i in range(n)]
    plt.xticks(ticks=np.arange(n), labels=labels, rotation=45, ha='right')
    plt.yticks(ticks=np.arange(n), labels=labels)

    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()

def select(data, idx):
    data = np.asarray(data)
    idx = np.asarray(idx)
    return data[:, idx]
