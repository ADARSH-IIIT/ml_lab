import numpy as np

class Perceptron:
    def __init__(self, w1, w2, bias, lr, threshold):
        self.w1 = w1
        self.w2 = w2
        self.bias = bias
        self.lr = lr
        self.threshold = threshold

    def activation(self, x):
        return 1 if x >= self.threshold else 0

    def predict(self, x1, x2):
        net = x1*self.w1 + x2*self.w2 + self.bias
        return self.activation(net)

    def train(self, X, y):
        epoch = 1
        while True:
            print(f"\nEpoch {epoch}:")
            errors = 0

            for i in range(len(X)):
                x1, x2 = X[i]
                target = y[i]

                output = self.predict(x1, x2)
                error = target - output

                if error != 0:
                    # weight update rule
                    self.w1 += self.lr * error * x1
                    self.w2 += self.lr * error * x2
                    self.bias += self.lr * error
                    errors += 1

                print(f" Input: {x1, x2}, Target: {target}, Pred: {output}, Error: {error}")
                print(f" Updated weights: w1={self.w1:.2f}, w2={self.w2:.2f}, bias={self.bias:.2f}")

            if errors == 0:
                print("\nTraining converged!")
                break

            epoch += 1


# -----------------------------
# OR GATE TRAINING
# -----------------------------
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,1])

p = Perceptron(w1=0.7, w2=1.3, bias=0.5, lr=0.5, threshold=1)
p.train(X, y)
