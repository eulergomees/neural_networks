import numpy as np

inputs = np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1], [1, 1, 0]])
outputs = np.array([1, 1, 0, 0])
bias = 0
weights = np.array([0, 0, 0])
maxIterations = 1000
learningRate = 0.1


# Função degrau
def activation(z):
    return 1 if z > 0 else 0  # 0 = não / 1 = sim


# Treinamento
def training(maxIterations, inputs, outputs, bias, weights):
    for iteration in range(maxIterations):
        for i in range(len(inputs)):
            sample = inputs[i]
            target = outputs[i]

            z = np.dot(sample, weights) + bias
            prediction = activation(z)

            if prediction == target:
                pass
            else:
                delta = learningRate * (target - prediction) * sample
                weights = weights + delta
                bias = bias + (learningRate * (target - prediction) * 1)
    return weights, bias


weights, bias = training(maxIterations, inputs, outputs, bias, weights)

# Teste com novo valor aleatorio
test = np.random.randint(0, 2, 3)
print(f"Valores sorteados: {test}")


def newInput(newInput, weights, bias):
    return activation(np.dot(newInput, weights) + bias)


print(f"Valor retornado: {newInput(test, weights, bias)}")
