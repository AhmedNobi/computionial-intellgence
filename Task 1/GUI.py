from tkinter import *
from tkinter import ttk, StringVar
from Draw_Data import ReadData
from NN_Algo import PerceptronAlgo
import numpy as np
def getData():
    pass

class GUI:
    def __init__(self):
        self.root = Tk()
        self.root.geometry("380x380")
        self.root.title('Perceptron Algorithm')
        self.root.option_add("*Font","Helvetica  10 bold italic")
        self.padY = (10, 10)
        self.padX = 20
        self.features = ["X1 & X2", "X1 & X3", "X1 & X4", "X2 & X3", "X2 & X4", "X3 & X4"]
        self.classes = ["C1 & C2", "C2 & C3", "C1 & C3"]
        self.chClasses = StringVar(self.root)
        self.chClasses.set(self.classes[0])
        self.chFeatures = StringVar(self.root)
        self.chFeatures.set(self.features[0])

        self.learnRate = StringVar(self.root)
        self.epochsNo = StringVar(self.root)
        self.bias = IntVar(self.root)
        self.bias.set(0)
        self.checked = IntVar(self.root)

        self.X1_training = []
        self.X2_training = []
        self.X1_testing = []
        self.X2_testing = []
        self.classlable_training = []
        self.classlable_testing = []

        self.c1 = 0
        self.c2 = 0
        self.w1 = 0
        self.w2 = 0
        self.b = 0

        self.initComp()
        self.root.mainloop()
    def initComp(self):

        #features######################
        firstFeatureLbl = Label(self.root, text="Select Two Feature:").grid(row=1, column=0, padx=self.padX, pady=self.padY, sticky=W)
        #self.ffCBox = ttk.Combobox(self.root, textvariable=self.mfeature, values=self.features).grid(row=1, column=1, padx=self.padX, pady=self.padY )

        self.ffCBox = ttk.Combobox(self.root, values=self.features, textvariable=self.chFeatures).grid(row=1, column=1, padx=self.padX, pady=self.padY)

        #secFeatureLbl = Label(self.root, text="Second Feature: ").grid(row=2, column=0, padx=self.padX, pady=self.padY, sticky=W)
        #self.sfCBox = ttk.Combobox(self.root, textvariable=self.mfeature2, values=self.features).grid(row=2, column=1, padx=self.padX, pady=self.padY)


        #classes######################
        firstClassLbl = Label(self.root, text="Select Two Classes:").grid(row=2, column=0, padx=self.padX, pady=self.padY, sticky=W)
        #self.fcCBox = ttk.Combobox(self.root, textvariable=self.mclass, values=self.classes).grid(row=2, column=1, padx=self.padX, pady=self.padY)

        #secClassLbl = Label(self.root, text="Second Class: ").grid(row=4, column=0, padx=self.padX, pady=self.padY, sticky=W)
        #self.scCBox = ttk.Combobox(self.root, textvariable=self.mclass2, values=self.classes).grid(row=4, column=1, padx=self.padX, pady=self.padY)

        self.fcCBox = ttk.Combobox(self.root, textvariable=self.chClasses, values=self.classes).grid(row=2, column=1, padx=self.padX, pady=self.padY)


        #entries######################
        lRateLbl = Label(self.root, text="Enter Learning Rate:").grid(row=3, column=0, padx=self.padX, pady=self.padY, sticky=W)
        self.lRateEntry = Entry(self.root,textvariable=self.learnRate).grid(row=3, column=1, padx=self.padX , pady=self.padY)

        epochLbl = Label(self.root, text="Enter #Epochs:").grid(row=4, column=0, padx=self.padX, pady=self.padY, sticky=W)
        self.epochEntry = Entry(self.root, textvariable=self.epochsNo).grid(row=4, column=1, padx=self.padX, pady=self.padY)


        #bias########################

        biasCheckBox = Checkbutton(self.root, text="Bias ",variable=self.bias).grid(row=5, column=0,padx=self.padX ,pady=self.padY, sticky=E)
        calButton = Button(self.root, text="All Combinations", bg="light Gray", command=lambda: self.possibleCombinations()).grid(row=5, column=1,padx=self.padX ,pady=self.padY, sticky=E)
        lrnButton = Button(self.root, text="Perceptron Algorithm", bg="light Gray", command=lambda: self.TrainingPhase()).grid(row=6, column=1, padx=self.padX, pady=self.padY, sticky=E)
        discriminativeButton = Button(self.root, text="Discriminative Features", bg="light Gray", command=lambda: self.plotFeatures()).grid(row=7,column=1,padx=self.padX,pady=self.padY,sticky=E)
        tstButton = Button(self.root, text="Testing Phase", bg="light Gray", command=lambda: self.TestingPhase()).grid(row=8, column=1, padx=self.padX, pady=self.padY, sticky=E)

    def possibleCombinations(self):
        rd = ReadData()
        rd.draw_iris_dataset()

    def plotFeatures(self):
        featureX, featureY,plt = self.initilizeData()
        if self.w1 !=0 and self.w2 != 0:
            class_ = [int(self.chClasses.get()[1]), int(self.chClasses.get()[6])]
            feature_ = [int(self.chFeatures.get()[1]), int(self.chFeatures.get()[6])]
            plt.discriminative_line(featureX, featureY, class_, feature_, self.w1, self.w2, self.b)

    def initilizeData(self):
        chosenFeatures = self.chFeatures.get()
        feature1 = int(chosenFeatures[1])
        feature2 = int(chosenFeatures[6])
        rd = ReadData()
        rd.readData()
        featureX = self.returnFeature(feature1,rd)
        featureY = self.returnFeature(feature2,rd)
        return (featureX, featureY,rd)

    def returnFeature(self,index,rd):
        feature = []
        if index == 1:
            feature = rd.IrisX1
        elif index == 2:
            feature = rd.IrisX2
        elif index == 3:
            feature = rd.IrisX3
        else:
            feature = rd.IrisX4
        return feature

    def manageTrainingFeatures(self):
        # initilize X1 and X2
        featureX, featureY,rd_ = self.initilizeData()
        choosenClasses = self.chClasses.get()
        class1 = int(choosenClasses[1])
        class2 = int(choosenClasses[6])
        self.classlable_training = []
        self.X1_training = []
        self.X2_training = []
        ####
        if class1 == 1:
            self.X1_training = featureX[0:30]
            self.X2_training = featureY[0:30]
            self.classlable_training = [1 for i in range(0, 30)]
            if class2 == 2:
                self.X1_training.extend(featureX[50:80])
                self.X2_training.extend(featureY[50:80])
                self.classlable_training.extend([2 for i in range(0, 30)])
            else:
                self.X1_training.extend(featureX[100:130])
                self.X2_training.extend(featureY[100:130])
                self.classlable_training.extend([3 for i in range(0, 30)])
        elif class1 == 2:
            self.X1_training = featureX[50:80]
            self.X2_training = featureY[50:80]
            self.classlable_training = [2 for i in range(0, 30)]
            self.X1_training.extend(featureX[100:130])
            self.X2_training.extend(featureY[100:130])
            self.classlable_training.extend([3 for i in range(0, 30)])
    def manageTestingFeatures(self):
        featureX, featureY,rd_ = self.initilizeData()
        choosenClasses = self.chClasses.get()
        class1 = int(choosenClasses[1])
        class2 = int(choosenClasses[6])
        self.c1 = class1
        self.c2 = class2
        ####
        if class1 == 1:
            self.X1_testing = featureX[30:50]
            self.X2_testing = featureY[30:50]
            self.classlable_testing = [1 for i in range(0, 20)]
            if class2 == 2:
                self.X1_testing.extend(featureX[80:100])
                self.X2_testing.extend(featureY[80:100])
                self.classlable_testing.extend([2 for i in range(0, 20)])
            else:
                self.X1_testing.extend(featureX[130:150])
                self.X2_testing.extend(featureY[130:150])
                self.classlable_testing.extend([3 for i in range(0, 20)])
        elif class1 == 2:
            self.X1_testing = featureX[80:100]
            self.X2_testing = featureY[80:100]
            self.classlable_testing = [2 for i in range(0, 20)]
            self.X1_testing.extend(featureX[130:150])
            self.X2_testing.extend(featureY[130:150])
            self.classlable_testing.extend([3 for i in range(0, 20)])

    def TrainingPhase(self):
        perceptron = PerceptronAlgo()
        bias = self.bias.get()
        eta = float(self.learnRate.get())
        epochsNo = int(self.epochsNo.get())
        self.manageTrainingFeatures()
        #print(self.X1_training, self.X2_training, self.classlable_training, eta, epochsNo, bias)
        w1,w2,b = perceptron.perceptronAlgoTrain(self.X1_training,self.X2_training,self.classlable_training,eta,epochsNo,bias)
        self.w1 = w1
        self.w2 = w2
        self.b = b
    def TestingPhase(self):
        perceptron = PerceptronAlgo()
        self.manageTestingFeatures()
        w = [self.w1,self.w2]
        Accuracy = 0
        conf = np.zeros([2,2], dtype='int32')
        print('-------------------------------------------------------------------------')
        for i in range(0,len(self.X1_testing)):
            y = perceptron.perceptronAlgoTest(self.X1_testing[i],self.X2_testing[i],w,self.b)
            if (y == 1 and self.c1 == self.classlable_testing[i]):
                conf[0, 0] = conf[0, 0] + 1
                Accuracy += 1
            elif (y == -1 and self.classlable_testing[i] == self.c2):
                conf[1,1] = conf[1,1] + 1
                Accuracy += 1
            elif (y == 1 and self.classlable_testing[i] != self.c1):
                conf[0, 1] = conf[0, 1] + 1
            elif (y == -1 and self.classlable_testing[i] != self.c2):
                conf[1, 0] = conf[1, 0] + 1
            print('X1: ' + str(self.X1_testing[i]), 'X2: ' + str(self.X2_testing[i]))
            print('Y: ' + str(y))
        print('classes: ' + str(self.classlable_testing))
        print('confusion matrix: ' + str(conf))
        print('Accuracy: ' + str((Accuracy/40)*100) +'%')