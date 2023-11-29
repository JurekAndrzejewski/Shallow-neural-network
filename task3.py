import sys

import math
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QLineEdit, QListWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIntValidator
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

import qdarktheme

np.set_printoptions(suppress=True)
plt.style.use('dark_background')

class NeuronLayer:
    
    def __init__(self, activationFunction, neuronAmountInput, neuronAmountOutput, trainingData, labels, first = False, parent=None, child=None):
        print("Neuron layer init")
        self.learningRate = 0.0001
        self.beta = -4
        self.parent = parent
        self.child = child
        self.trainingData = trainingData
        self.method = activationFunction
        self.neuronAmountInput = neuronAmountInput
        self.neuronAmountOutput = neuronAmountOutput
        if first:
            self.trainingData = np.c_[self.trainingData, np.ones(np.shape(self.trainingData)[0])]
        self.weights = np.random.rand(neuronAmountInput, neuronAmountOutput)
        self.labels = labels.reshape(-1, 1)
        self.calculateOutput()
        
    def forward(self, x):
        self.state = x @ self.weights
        self.output = self.methodPicker(self.state)
        self.derivativeStates = self.derivativePicker(self.state)
        return self.output
    
    def backward(self):
        if self.child is None:
            self.error = self.output - self.labels
            self.derivativeStates = self.derivativePicker(self.state)
            self.delta = self.calculateDeltaWeight(last = True)
        else:
            self.derivativeStates = self.derivativePicker(self.state)
            self.delta = self.calculateDeltaWeight(last = False)
        return self.delta
    
    def calculateDeltaWeight(self, last: bool):
        if last:
            self.delta = (self.error * self.derivativeStates)
        else:
            self.delta = self.child.delta @ self.child.weights.T * self.derivativeStates
        return self.delta
    
    def update(self, input):
        updateValue = input.T.dot(self.delta)
        self.weights -= self.learningRate * updateValue
        
    def calculateOutput(self):
        self.state = self.trainingData @ self.weights
        self.output = self.methodPicker(self.state)
        return self.output    
    
    ##############################################
    # ACTIVATION FUNCTION AND DERIVATIVE PICKERS #
    ##############################################
    
    def methodPicker(self, output):
        if self.method == 1:
            #sigmoid
            return self.logistic(output)
        elif self.method == 2:
            #sin
            return np.sin(output)
        elif self.method == 3:
            #tanh
            return np.tanh(output)
        elif self.method == 4:
            #sign
            return np.sign(output)
        elif self.method == 5:
            #relu
            return self.relu(output)
        elif self.method == 6:
            #drelu
            return self.leakyRelu(output)
        else: # default method
            return np.heaviside(output, 0.5)
        
    def derivativePicker(self, output):
        if self.method == 1:
            #sigmoid
            value = self.logistic(output) * (1 - self.logistic(output))
            return value
        elif self.method == 2:
            #sin
            return np.cos(output)
        elif self.method == 3:
            #tanh
            return (1 - np.tanh(output) * np.tanh(output))
        elif self.method == 4:
            #sign
            return np.ones(output.shape)
        elif self.method == 5:
            #relu
            return self.reluD(output)
        elif self.method == 6:
            #lrelu
            return self.leakyReluD(output)
        else: #default is derivative for heaviside
            return np.ones(output.shape)
        
    #############################
    # FUNCTIONS AND DERIVATIVES #
    #############################
    
    def logistic(self, output):
        return 1.0 / (1 + np.exp(self.beta * output))
    
    def relu (self, output):
        return np.where(output > 0, output, 0)

    def reluD (self, output):
        return np.where(output > 0, 1, 0)

    def leakyRelu(self, output):
        return np.where(output > 0, output, 0.01 * output)

    def leakyReluD(self, output):
        return np.where(output > 0, 1, 0.01 * output)
    
class NeuronNetwork:
    def __init__(self, set, activationFunction, layerAmount, neuronAmount):
        print("Neuron network init")
        self.layerAmount = layerAmount
        self.epochs = 10000
        self.labels, self.trainingData = self.extractLabel(set)
        self.method = activationFunction
        self.layerList = [NeuronLayer(activationFunction, 3, 3, self.trainingData, self.labels, first = True)]
        for i in range(1, layerAmount-1):
            self.layerList.append(NeuronLayer(activationFunction, len(self.layerList[i-1].output[0]), neuronAmount+1, self.layerList[i-1].output, self.labels, parent = self.layerList[i-1]))
            self.layerList[i-1].child = self.layerList[i]

        self.layerList.append(NeuronLayer(activationFunction, len(self.layerList[layerAmount-2].output[0]), 2, self.layerList[layerAmount-2].output, self.labels, first = False, parent = self.layerList[layerAmount-2]))
        self.layerList[layerAmount-2].child = self.layerList[layerAmount-1]
    
    def extractLabel(self, trainingData):
        rowLength = len(trainingData[0])
        labels = trainingData[:, rowLength - 1]
        trainingData = np.delete(trainingData, rowLength - 1, 1)
        return labels, trainingData
    
    def train(self):
        trainInput = np.c_[self.trainingData, np.ones(np.shape(self.trainingData)[0])]
        for i in range(self.epochs):
            input = trainInput
            for layer in self.layerList:
                input = layer.forward(input)
            self.layerList[self.layerAmount-1].backward()
            for i in reversed(range(self.layerAmount)):
                self.layerList[i].backward()
            input = trainInput
            for layer in self.layerList:
                layer.update(input)
                input = layer.output
    
    def preparePointsToPredict(self, x, y):
        input = np.c_[x.reshape(-1, 1), y.reshape(-1, 1)]
        bias = (np.ones(np.shape(input)[0]))
        input = np.c_[input, bias.reshape(-1, 1)]
        for layer in self.layerList:
            input = layer.forward(input)
        return input
    
    
    
class Plot(FigureCanvasQTAgg):
    def __init__(self, parent = None):
        self.plot = plt.figure()
        self.plot.set_facecolor("#202124")
        self.scatter = self.plot.add_subplot(111)
        super().__init__(self.plot)
        self.setParent(parent)
        self.createPlot(50, 2)
    
    def createSet(self, class0Samples, class1Samples):
        class0_temp = class0Samples[0]
        class1_temp = class1Samples[0]
        for i in range(1, class0Samples.shape[0]):
            class0_temp = np.r_[class0_temp, class0Samples[i]]
            class1_temp = np.r_[class1_temp, class1Samples[i]]
        
        shape0 = class0_temp.shape[0] + class1_temp.shape[0]
        temp = np.zeros((shape0, 3))
        for x in range(class0_temp.shape[0]):
            temp[x] = np.array([class0_temp[x][0], class0_temp[x][1], 0])
            temp[x+int(shape0/2)] = np.array([class1_temp[x][0], class1_temp[x][1], 1])

        np.random.shuffle(temp)
        self.set = temp
        
    def getSet(self):
        return self.set
    
    def createPlot(self, samplesNumber = 50, modesNumber = 2):
        self.plot.clear()
        self.scatter = self.plot.add_subplot(111)
        self.scatter.set_facecolor("#202124")
        class0_modes = np.random.rand(modesNumber, 2)
        class1_modes = np.random.rand(modesNumber, 2)
        class0Samples = np.zeros((modesNumber, samplesNumber, 2))
        class1Samples = np.zeros((modesNumber, samplesNumber, 2))
        
        for i in range(modesNumber):
            class0Samples[i] = np.random.normal(loc = class0_modes[i], scale = 0.1, size = (samplesNumber, 2))
            class1Samples[i] = np.random.normal(loc = class1_modes[i], scale = 0.1, size = (samplesNumber, 2))

        self.scatter.scatter(class0_modes[:, 0], class0_modes[:, 1], c = 'red', marker = "P")
        self.scatter.scatter(class0Samples[:, :, 0], class0Samples[:, :, 1], c = 'red', marker = '*')
        self.scatter.scatter(class1_modes[:, 0], class1_modes[:, 1], c = 'blue', marker = "P")
        self.scatter.scatter(class1Samples[:, :, 0], class1Samples[:, :, 1], c = 'blue', marker = '*')
        
        self.class0Samples = class0Samples
        self.class1Samples = class1Samples
        
        self.createSet(class0Samples, class1Samples)
        self.getMaxes()
        self.scatter.set_xlim(self.minX, self.maxX)
        self.scatter.set_ylim(self.minY, self.maxY)
    
    def getMaxes(self):
        self.minX = np.amin(np.hstack(self.set[:,0])) * 1.1
        self.maxX = np.amax(np.hstack(self.set[:,0])) * 1.1
        
        self.minY = np.amin(np.hstack(self.set[:,1])) * 1.1
        self.maxY = np.amax(np.hstack(self.set[:,1])) * 1.1
        
    def createBoundary(self): 
        
        xAxis = np.linspace(self.minX, self.maxX, 50)
        yAxis = np.linspace(self.minY, self.maxY, 50)
        
        xVals, yVals = np.meshgrid(xAxis, yAxis)
        
        return xVals, yVals

    def createPlotWithBoundary(self, prediction, x, y):
        self.plot.clear()
        self.scatter = self.plot.add_subplot(111)
        self.scatter.set_facecolor("#202124")
        self.scatter.set_xlim(self.minX, self.maxX)
        self.scatter.set_ylim(self.minY, self.maxY)
        self.scatter.contourf(x, y, prediction, alpha = 0.5)#, cmap=cm.jet)
        self.scatter.scatter(self.class0Samples[:, :, 0], self.class0Samples[:, :, 1], c = 'red', marker = '*')
        self.scatter.scatter(self.class1Samples[:, :, 0], self.class1Samples[:, :, 1], c = 'blue', marker = '*')
        
        
class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.plot = Plot(self)
        self.setGeometry(50,50,900,480)
        
        onlyInt = QIntValidator()
        onlyInt.setRange(1, 1000)

        self.modesLabel = QLabel("number of modes", self)
        self.modesLabel.setGeometry(710, 30, 100, 20)
        
        self.numberOfModesWidget = QLineEdit(self)
        self.numberOfModesWidget.setText("2")
        self.numberOfModesWidget.setGeometry(810, 30, 50, 20)
        self.numberOfModesWidget.setValidator(onlyInt)
        
        self.samplesLabel = QLabel("number of samples", self)
        self.samplesLabel.setGeometry(710, 60, 100, 20)
        
        self.numberOfSamplesWidget = QLineEdit(self)
        self.numberOfSamplesWidget.setText("50")
        self.numberOfSamplesWidget.setGeometry(810, 60, 50, 20)
        self.numberOfSamplesWidget.setValidator(onlyInt)
        
        self.createPlotButton = QPushButton(self)
        self.createPlotButton.setText("create plot")
        self.createPlotButton.setGeometry(720, 90, 130, 20)
        self.createPlotButton.clicked.connect(lambda: self.createPlot())
        
        self.methodWidget = QListWidget(self)
        self.methodWidget.addItem('Heaviside')
        self.methodWidget.addItem('Logistic')
        self.methodWidget.addItem('Sin')
        self.methodWidget.addItem('Tanh')
        self.methodWidget.addItem('Sign')
        self.methodWidget.addItem('Relu')
        self.methodWidget.addItem('LRelu')
        self.methodWidget.setGeometry(735, 120, 100, 140)
        
        self.layersLabel = QLabel("number of layers", self)
        self.layersLabel.setGeometry(710, 270, 100, 20)
        
        self.numberOfLayersWidget = QLineEdit(self)
        self.numberOfLayersWidget.setText("3")
        self.numberOfLayersWidget.setGeometry(810, 270, 50, 20)
        self.numberOfLayersWidget.setValidator(onlyInt)
        
        self.neuronsLabel = QLabel("number of neurons", self)
        self.neuronsLabel.setGeometry(710, 300, 100, 20)
        
        self.numberOfNeuronsWidget = QLineEdit(self)
        self.numberOfNeuronsWidget.setText("5")
        self.numberOfNeuronsWidget.setGeometry(810, 300, 50, 20)
        self.numberOfNeuronsWidget.setValidator(onlyInt)
        
        self.neuronButton = QPushButton(self)
        self.neuronButton.setText("Create Neural Network")
        self.neuronButton.setGeometry(720, 330, 130, 20)
        self.neuronButton.clicked.connect(lambda: self.createNeuralNetwork())
        
        self.neuronTrainButton = QPushButton(self)
        self.neuronTrainButton.setText("Train Neural Network")
        self.neuronTrainButton.setGeometry(720, 360, 130, 20)
        self.neuronTrainButton.clicked.connect(lambda: self.trainNeuralNetwork())
        
        self.show()

    def createPlot(self):
        self.plot.createPlot(modesNumber=int(self.numberOfModesWidget.text()), 
                             samplesNumber=int(self.numberOfSamplesWidget.text()))
        self.plot.draw()
        
    def createNeuralNetwork(self):
        if int(self.numberOfLayersWidget.text()) > 5:
            self.numberOfLayersWidget.setText("5")
        elif int(self.numberOfLayersWidget.text()) < 3:
            self.numberOfLayersWidget.setText("3")
        self.neuronNetwork = NeuronNetwork(self.plot.getSet(), int(self.methodWidget.currentRow()), int(self.numberOfLayersWidget.text()), int(self.numberOfNeuronsWidget.text()))
    
    def trainNeuralNetwork(self):
        self.neuronNetwork.train()
        self.xVals, self.yVals = self.plot.createBoundary()
        self.predictedLabels = self.neuronNetwork.preparePointsToPredict(self.xVals, self.yVals)
        self.predictedLabels = self.predictedLabels[:, 0].reshape(self.xVals.shape)
        
        self.plot.createPlotWithBoundary(self.predictedLabels, self.xVals, self.yVals)
        self.plot.draw()
        

def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarktheme.load_stylesheet())
    
    window = Window()
    window.show()

    app.exec()
    
if __name__ == "__main__":
    main()
        