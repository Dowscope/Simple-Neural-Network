import NeuralNetwork

nn = NeuralNetwork.NeuralNetwork(2,4,1,0.01)

# Train the Network
inputs = [
    [0,0],
    [0,1],
    [1,0],
    [1,1]
]

targets = [0,1,1,0]

for i in range(100000):
    for j in range(4):
        nn.train(inputs[j], targets[j])
        #print(f'Input: {inputs[j]}, target: {targets[j]}')


print(nn.query([0,1]))

print(nn.query([0,0]))
print(nn.query([1,1]))
print(nn.query([1,0]))