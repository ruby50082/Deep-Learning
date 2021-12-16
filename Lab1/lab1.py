import numpy as np
import matplotlib.pyplot as plt

def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0]-pt[1])/1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)

def generate_XOR_easy():
    inputs = []
    labels = []
    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)
        
        if 0.1*i == 0.5:
            continue
        
        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)

    return np.array(inputs), np.array(labels).reshape(21, 1)

def show_result(x, y, y_pred):
    plt.subplot(1,2,1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
            
    plt.subplot(1,2,2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if y_pred[i] <= 0.5:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
        
    plt.show()

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def derivative_sigmoid(x):
    return np.multiply(x, 1.0 - x)

def forward(x, w1, w2, w3):
    a1 = np.dot(x, w1)
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2)
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3)
    y_pred = sigmoid(a3)
    
    return a1, z1, a2, z2, a3, y_pred

def backward(x, y, y_pred, w1, w2, w3, z1, z2):
    sigma3 = y_pred - y
    w3 -= lr * np.dot(z2.T, sigma3)

    sigma2 = np.dot(sigma3, w3.T) * derivative_sigmoid(z2)
    w2 -= lr * np.dot(z1.T, sigma2)

    sigma1 = np.dot(sigma2, w2.T) * derivative_sigmoid(z1)
    w1 -= lr * np.dot(x.T, sigma1)

    return w1, w2, w3

def train(x, y, w1, w2, w3):
    print('Training:')
    loss_list = []
    for epoch in range(10001):
        a1, z1, a2, z2, a3, y_pred = forward(x, w1, w2, w3)
        w1, w2, w3 = backward(x, y, y_pred, w1, w2, w3, z1, z2)
        loss = np.mean(np.square(y - y_pred))
        loss_list.append(loss)
        if epoch % 1000 == 0:
            print('epoch {} loss : {:f}'.format(epoch, loss))

    epochs = range(0, 10001)
    plt.plot(epochs, loss_list, 'r')
    plt.title('Learning Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

    return w1, w2, w3

def test(x, y, w1, w2, w3):
    print('Testing:')
    a1, z1, a2, z2, a3, y_pred = forward(x, w1, w2, w3)
    print('Prediction:')
    np.set_printoptions(suppress=True)
    print('{}'.format(y_pred))

    y_pred_final = [[0 if y_pred[i] <= 0.5 else 1 for i in range(y_pred.shape[0])]]
    acc = np.mean(y == np.array(y_pred_final).T) * 100
    print('Accuracy: {:.2f}%'.format(acc))
    show_result(x, y, y_pred)


if __name__ == "__main__":
    lr = 0.05
    hidden_size = 4

    x, y = generate_linear(n=100)
    w1 = np.random.randn(2, hidden_size)
    w2 = np.random.randn(hidden_size, hidden_size)
    w3 = np.random.randn(hidden_size, 1)
    w1, w2, w3 = train(x, y, w1, w2, w3)
    test(x, y, w1, w2, w3)

    x, y = generate_XOR_easy()
    w1 = np.random.randn(2, hidden_size)
    w2 = np.random.randn(hidden_size, hidden_size)
    w3 = np.random.randn(hidden_size, 1)
    w1, w2, w3 = train(x, y, w1, w2, w3)
    test(x, y, w1, w2, w3)
