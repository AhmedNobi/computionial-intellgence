from tkinter import *
from tkinter import ttk, StringVar
from Logic import Logic_management

def getData():
    pass

class GUI:
    def __init__(self):
        self.root = Tk()
        self.root.geometry("400x400")
        self.root.title('Adaline Algorithm')
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
        self.threshold = StringVar(self.root)
        self.bias = IntVar(self.root)
        self.bias.set(0)
        self.checked = IntVar(self.root)
        self.initComp()

        self.manage = Logic_management()

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

        thresholdLbl = Label(self.root, text="Enter MSE Threshold:").grid(row=5, column=0, padx=self.padX,
                                                                         pady=self.padY, sticky=W)
        self.thresholdEntry = Entry(self.root, textvariable=self.threshold).grid(row=5, column=1, padx=self.padX,
                                                                                 pady=self.padY)
        #bias########################

        biasCheckBox = Checkbutton(self.root, text="Bias ", variable=self.bias).grid(row=6, column=0, padx=self.padX,
                                                                                     pady=self.padY, sticky=E)
        calButton = Button(self.root, text="All Combinations", bg="light Gray",
                           command=lambda: self.manage.possibleCombinations()).grid(row=6, column=1, padx=self.padX,
                                                                             pady=self.padY, sticky=E)
        lrnButton = Button(self.root, text="Adaline Algorithm", bg="light Gray",
                           command=lambda: self.manage.TrainingPhase(self.bias.get(),  float(self.learnRate.get()),
                                                                     int(self.epochsNo.get()), self.chFeatures.get(),
                                                                     self.chClasses.get(), float(self.threshold.get()))).grid(row=7, column=1, padx=self.padX, pady=self.padY,
                                                                 sticky=E)
        discriminativeButton = Button(self.root, text="Discriminative Features", bg="light Gray",
                                      command=lambda: self.manage.plotFeatures()).grid(row=9, column=1, padx=self.padX,
                                                                                pady=self.padY, sticky=E)
        tstButton = Button(self.root, text="Testing", bg="light Gray", command=lambda: self.manage.TestingPhase()).grid(row=8,
                                                                                                            column=1,
                                                                                                            padx=self.padX,
                                                                                                            pady=self.padY,
                                                                                                            sticky=E)
