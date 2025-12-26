import numpy as np

# activatio layer
def relu(z):
    return np.maximum(0, z)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# ham loss
def cross_entropy(y, y_hat):
    eps = 1e-8
    return -np.mean(
        y * np.log(y_hat + eps) +
        (1 - y) * np.log(1 - y_hat + eps)
    )
def init_para(layers):
    para = {}
    for i in range(1, len(layers)):
        para[f"W{i}"] = np.random.randn(layers[i-1], layers[i]) * 0.01
        para[f"b{i}"] = np.zeros((1, layers[i]))
    return para
def forward(X, para):
    cache = {"A0": X}
    L = len(para) // 2

    for i in range(1, L):
        Z = cache[f"A{i-1}"] @ para[f"W{i}"] + para[f"b{i}"]
        A = relu(Z)
        cache[f"Z{i}"], cache[f"A{i}"] = Z, A

    ZL = cache[f"A{L-1}"] @ para[f"W{L}"] + para[f"b{L}"]
    AL = sigmoid(ZL)

    cache[f"Z{L}"], cache[f"A{L}"] = ZL, AL
    return cache
def backward_multi(X, Y, para, cache):
    grads = {}
    m = X.shape[0]

    L = len(para) // 2


    dZ = cache[f"A{L}"] - Y
    grads[f"dW{L}"] = cache[f"A{L-1}"].T @ dZ / m
    grads[f"db{L}"] = np.sum(dZ, axis=0, keepdims=True) / m

    # ===== Hidden layers (ngược lại) =====
    for i in reversed(range(1, L)):
        dA = dZ @ para[f"W{i+1}"].T
        dZ = dA * (cache[f"Z{i}"] > 0)

        A_prev = cache["A0"] if i == 1 else cache[f"A{i-1}"]

        grads[f"dW{i}"] = A_prev.T @ dZ / m
        grads[f"db{i}"] = np.sum(dZ, axis=0, keepdims=True) / m

    return grads
def update_params_multi(para, grads, lr):
    L = len(para) // 2
    for i in range(1, L + 1):
        para[f"W{i}"] -= lr * grads[f"dW{i}"]
        para[f"b{i}"] -= lr * grads[f"db{i}"]
    return para
def train_model(X, y ,  layers, lr, epochs=1000, verbose=False):
    para = init_para(layers)

    for epoch in range(epochs):
        # Forward
        cache = forward(X, para)

        # Loss
        loss = cross_entropy(y, cache[f"A{len(layers)-1}"])

        # Backward + update
        grads = backward_multi(X, y, para, cache)
        para = update_params_multi(para, grads, lr)


    return para

# ham du doan
def predict(X, para, threshold=0.5):

    cache = forward(X, para)
    y_hat = cache[f"A{len(para)//2}"]  # output layer
    return (y_hat >= threshold).astype(int)

def accuracy(y_true, y_pred):
    assert y_true.shape == y_pred.shape, \
        f"Shape mismatch: {y_true.shape} vs {y_pred.shape}"

    return np.mean(y_true == y_pred)


