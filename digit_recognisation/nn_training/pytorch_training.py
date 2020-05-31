import numpy as np
import torchvision
import torch
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
from digit_recognition.nn_training.util import view_classify


# define transformations to be performed on testdata before feeding into pipeline
# ToTensor converts image into tensor of rgb values scaled to number between 0 and 1
# Normalize normalizes tensor with mean and standard deviation
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# download dataset
trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
valset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)
# shuffle dataset and load to DataLoader.
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

# create iterator over trainloader. Extract images and labels from iterator
dataiter = iter(trainloader)
images, labels = dataiter.next()
# print size and shape of images and labels
"""print(images.shape)
print(labels.shape)"""

# display images
"""figure = plt.figure()
num_of_images = 60
for index in range(1, num_of_images + 1):
    plt.subplot(6, 10, index)
    plt.axis('off')
    plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')
    plt.show()"""



"""----------------CREATE NEURAL NETWORK----------------"""
# define input size = 28x28=784, hidden layer size and outputlayer size
input_size = 784
hidden_sizes = [128, 64]
output_size = 10

# Wrap layes of the network using nn.
# 3 linear layers weid ReLU activation which lets positive values pass through
# and sets negative values to zero.
# Output layer is linear layer with logarithm of softmax function activation
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()
images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)

logps = model(images) # log probabilities
loss = criterion(logps, labels) # caluclate NLL loss

# print model weights before and after loss.backward() updates the weights of the network to minimize loss
"""print('Before backward pass: \n', model[0].weight.grad)"""
loss.backward()
"""print('After backward pass: \n', model[0].weight.grad)"""




"""----------------TRAIN NETWORK----------------"""
# initialize optimizer
optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
time0 = time()

# epoch = # of times training set is iterated over
# in each epoch perform gradient descent and update weights by back-propogation
epochs = 15
for epoch in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # flatten MNIST images into vector of size 784
        images = images.view(images.shape[0], -1)

        # training pass
        optimizer.zero_grad()

        output = model(images)
        loss = criterion(output, labels)

        # let model learn by backpropogating
        loss.backward()

        # optimize model weights
        optimizer.step()

        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(epoch, running_loss/len(trainloader)))

print("\nTraining time (in minutes) = ", (time()-time0)/60)


# Display an image and the models perceived probability of what number is displayed
images, labels = next(iter(valloader))

img = images[0].view(1, 784)

with torch.no_grad():
    logps = model(img)

ps = torch.exp(logps)
probab = list(ps.numpy()[0])
print("Predicted Digit = ", probab.index(max(probab)))
view_classify(img.view(1, 28, 28), ps)


# Iterate through validation set and calculate total correct predictions
correct_count, all_count = 0, 0
for images, labels in valloader:
    for i in range(len(labels)):
        img = images[i].view(1, 784)
        with torch.no_grad():
            logps = model(img)

        ps = torch.exp(logps)
        probab = list(ps.numpy()[0])
        pred_label = probab.index(max(probab))
        true_label = labels.numpy()[i]
        if (true_label == pred_label):
            correct_count += 1
        all_count += 1

print("Number of images tested = ", all_count)
print("\nModel accuracy = ", (correct_count/all_count))

torch.save(model, './mnist_number_model.pt')

