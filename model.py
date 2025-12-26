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

# khoi tao trong so
def init_para(n_input,n_hidden ,n_output):
    para = {
        "w1" : np.random.randn(n_input,n_hidden) * 0.01,
        "b1" : np.zeros((1, n_hidden)),
        "w2" : np.random.rand(n_hidden, n_output) * 0.01,
        "b2" : np.zeros((1, n_output))
    }
    return para

# lan truyen tien
def forward(X, para):
    w1, b1, w2, b2 = para["w1"], para["b1"], para["w2"], para["b2"]
    z1 = np.dot(X, w1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1 , w2) +  b2
    a2 = sigmoid(z2)
    y_hat = a2
    cache = {
        "z1": z1 ,
        "a1": a1 ,
        "a2": a2 ,
        "z2": z2 ,
        "y_hat": y_hat

    }
    return cache

# lan truyen nguoc
def backward(X, Y, para, cache):
    m = X.shape[0]
    w2 = para["w2"]

    a1 = cache["a1"]
    a2 = cache["a2"]
    z1 = cache["z1"]

    dz2 = cache["y_hat"] - Y
    dw2 = np.dot(a1.T, dz2) / m
    db2 = np.sum(dz2, axis=0, keepdims=True) / m

    dz1 = np.dot(dz2, w2.T) * (z1 > 0)
    dw1 = np.dot(X.T, dz1) / m
    db1 = np.sum(dz1, axis=0, keepdims=True) / m

    grads = {
        "dw1": dw1,
        "db1": db1,
        "dw2": dw2,
        "db2": db2
    }
    return grads

# cap nhat trong so
def update_params(para, grads, learning_rate):
    para["w1"] -= learning_rate * grads["dw1"]
    para["b1"] -= learning_rate * grads["db1"]
    para["w2"] -= learning_rate * grads["dw2"]
    para["b2"] -= learning_rate * grads["db2"]
    return para

def train(X, Y , epochs , learning_rate):
    n_input = X.shape[1]
    n_hidden = 8
    n_output = 1
    para = init_para(n_input, n_hidden, n_output)
    for i in range (epochs):
        cache = forward(X, para)
        grads = backward(X, Y, para, cache)
        loss = cross_entropy(Y, cache["y_hat"])
        para = update_params(para, grads, learning_rate)

        print(f"epoch: {i}, loss: {loss}")

    return para

# ham du doan
def predict(X, para):
    y_hat = forward(X, para)["y_hat"]
    return (y_hat >= 0.5).astype(int)
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


