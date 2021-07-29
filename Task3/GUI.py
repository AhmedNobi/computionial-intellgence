import tkinter as tk
from Logic import Logic

class GUI:
    def __init__(self):
        self.root = tk.Tk()

        self.learnRate = tk.StringVar(self.root)
        self.learnRate.set(0.01)
        self.epochsNo = tk.StringVar(self.root)
        self.epochsNo.set(4)

        self.bias = tk.IntVar(self.root)
        self.bias.set(0)

        self.root.resizable(height=True, width=True)
        self.root.title("Multi Layer Neural Networks - Task3")
        self.root.geometry("400x450")

        self.errorTh =tk.DoubleVar(self.root)
        self.errorTh.set(0.001)

        self.plotLine = tk.IntVar(self.root)
        self.plotLine.set(0)

        self.NumberOfHiddenLayers = tk.IntVar(self.root)
        self.NumberOfHiddenLayers.set(3)

        self.NumberOfNeuronsInEachLayer = tk.StringVar(self.root)
        self.NumberOfNeuronsInEachLayer.set("3,2,2")

        self.activationFunction = tk.IntVar()
        self.activationFunction.set(1)

        self.manage = Logic()

        self.stoppingCriteria = tk.IntVar()
        self.stoppingCriteria.set(1)
        self.initialComp()
        self.root.mainloop()
        #######################

    def initialComp(self):
        def defocus(event):
            event.widget.master.focus_set()
        tk.Label(self.root,text="Number Of Hidden Layers").place(relx=0.11, rely=0.05)
        HLEntry = tk.Entry(self.root,width=10 , textvariable = self.NumberOfHiddenLayers)
        HLEntry.place(relx=0.64, rely=0.05)

        tk.Label(self.root,text="Number Of Neurons In Each Layer").place(relx=0.11, rely=0.14)
        NEntry = tk.Entry(self.root, width=10, textvariable = self.NumberOfNeuronsInEachLayer)
        NEntry.place(relx=0.64, rely=0.14)

        tk.Label(self.root,text="Enter Learning Rate(eta):").place(relx=0.11, rely=0.23)
        eta = tk.Entry(self.root,width=10,textvariable=self.learnRate)
        eta.place(relx=0.64, rely=0.23)

        tk.Label(self.root,text="Enter Number of Epochs:").place(relx=0.11, rely=0.32)
        epochs = tk.Entry(self.root,width=10,textvariable=self.epochsNo)
        epochs.place(relx=0.64, rely=0.32)

        tk.Label(self.root, text="Choose an activation function").place(relx=0.03, rely=0.41)
        tk.Radiobutton(self.root, text="Sigmoid", variable=self.activationFunction, value=1).place(relx=0.03, rely=0.45)
        tk.Radiobutton(self.root, text="Hyperbolic Tangent Sigmoid", variable=self.activationFunction, value=2).place(relx=0.03, rely=0.49)

        tk.Label(self.root, text="Bias:").place(relx=0.70, rely=0.41)
        tk.Checkbutton(self.root, variable=self.bias).place(relx=0.78, rely=0.41)

        tk.Button(self.root, text="Learning",width=10, fg="Black",bg="light Gray",  command=lambda: self.manage.TrainingPhase(self.bias.get(),  float(self.learnRate.get()),
                                                                     int(self.epochsNo.get()),int(self.NumberOfHiddenLayers.get()), self.NumberOfNeuronsInEachLayer.get(), self.activationFunction.get())).place(relx=0.35, rely=0.80)
        tk.Button(self.root, text="Testing",width=10, fg="Black",bg="light Gray",  command=lambda: self.manage.TestingPhase()).place(relx=0.67, rely=0.80)
        

if __name__ == '__main__':
    app = GUI()
