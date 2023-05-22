import numpy as np
import torch
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


def read_data() -> tuple:
    data = []
    labels = np.array([])
    try:
        with open("data.txt") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                if line == '\n':
                    continue
                line_data_str = line.split(',')
                line_data_int = list(map(int, line_data_str))
                if len(line_data_int) != 101:
                    continue
                labels = np.append(labels, line_data_int[0])
                del line_data_int[0]
                data.append(np.array(line_data_int))
            f.close()
    except:
        # print("error in clean", i)
        pass
    return data, labels

def train_test_split(data, labels, num) -> tuple:
    if len(data) != len(labels):
        print("ERROR")
        exit(1)
    data_length = len(labels)
    X = torch.tensor(np.array(data))
    y = torch.tensor(np.array(labels).astype(int), dtype=torch.int)
    torch.manual_seed(42)
    shuffle_idx = torch.randperm(data_length, dtype=torch.long)
    X, y = X[shuffle_idx], y[shuffle_idx]
    data_size = shuffle_idx.size(0)

    percent80 = int(data_size * 0.8)
    X_train, X_test = X[shuffle_idx[:percent80]], X[shuffle_idx[percent80:]]
    y_train, y_test = y[shuffle_idx[:percent80]], y[shuffle_idx[percent80:]]
    return X_train, X_test, y_train, y_test


def run_backpropagation(first, second, data, labels):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, 1)

    train_indices = []
    test_indices = []

    if first == "b" and second == "m":
        train_indices = [i for i, y in enumerate(y_train) if y == 1 or y == 3]
        test_indices = [i for i, y in enumerate(y_test) if y == 1 or y == 3]
    elif first == "b" and second == "l":
        train_indices = [i for i, y in enumerate(y_train) if y == 1 or y == 2]
        test_indices = [i for i, y in enumerate(y_test) if y == 1 or y == 2]
    elif first == "l" and second == "m":
        train_indices = [i for i, y in enumerate(y_train) if y == 2 or y == 3]
        test_indices = [i for i, y in enumerate(y_test) if y == 2 or y == 3]

    X_train_filtered = X_train[train_indices]
    y_train_filtered = y_train[train_indices]

    X_test_filtered = X_test[test_indices]
    y_test_filtered = y_test[test_indices]

    mlp = MLPClassifier(hidden_layer_sizes=(100, 100), activation='relu', solver='adam', random_state=42, max_iter=1000)
    mlp.fit(X_train_filtered, y_train_filtered)
    y_pred = mlp.predict(X_test_filtered)
    accuracy = accuracy_score(y_test_filtered, y_pred)
    print("Accuracy:", accuracy)


if __name__ == '__main__':
    data, labels = read_data()
    print("------------------------")
    print("------------------------")
    print("bet vs lamed:")
    run_backpropagation('b', 'l', data, labels)
    print("------------------------")
    print("------------------------")
    print("bet vs mem:")
    run_backpropagation('b', 'm', data, labels)
    print("------------------------")
    print("------------------------")
    print("lamed vs mem:")
    run_backpropagation('l', 'm', data, labels)
    print("------------------------")
    print("------------------------")


