from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
from Backpropagation import backPrpagation
class Logic:
    def __init__(self):
        self.IrisX1 = []
        self.IrisX2 = []
        self.IrisX3 = []
        self.IrisX4 = []
        self.IrisY = []

        self.training_data = []
        self.training_label = []

        self.testing_data = []
        self.testing_label = []
        ################ Input Data ####################
        self.bias = 0
        self.learnRate = 0
        self.numOfEpochs = 0
        self.Num_Of_Hdn_Lyrs =0
        self.Num_Of_Nern_Ech_Lyr =[]
        self.Activation_is_Sigmoid = 0
        ################################################3

    def readData(self):
        with open('IrisData.txt') as irisTxt:
            irisTxt.readline()
            for x in irisTxt:
                X = x.split(',')
                self.IrisX1.append(float(X[0]))
                self.IrisX2.append(float(X[1]))
                self.IrisX3.append(float(X[2]))
                self.IrisX4.append(float(X[3]))
                self.IrisY.append(X[4].splitlines()[0])
        return self.IrisX1, self.IrisX2, self.IrisX3, self.IrisX4, self.IrisY

    def preprocessing_data(self):
        #get integer classes
        Label_encoder = LabelEncoder()
        classes_y = Label_encoder.fit_transform(self.IrisY)
        classes_y = classes_y.reshape(len(classes_y) ,1) #[150,1]

        # binary classes
        one_hot_encoder = OneHotEncoder(sparse=False, categories='auto')
        enoder_y = one_hot_encoder.fit_transform(classes_y)
        self.IrisY = enoder_y
        return self.IrisX1, self.IrisX2, self.IrisX3, self.IrisX4, self.IrisY

    def FeaturesSplit(self):
        #################################3
        self.readData()
        self.preprocessing_data()

        #self.IrisX1  = np.asarray(self.IrisX1) /max(self.IrisX1)
        #self.IrisX2  = np.asarray(self.IrisX2) /max(self.IrisX2)
        #self.IrisX3  = np.asarray(self.IrisX3) / max(self.IrisX3)
        #self.IrisX4  = np.asarray(self.IrisX4) /max(self.IrisX4)
        ##################### Training ###############################
        for i in range(0,150,50):
            for x in range(i, i+30):
                self.training_data.extend([[self.IrisX1[x], self.IrisX2[x], self.IrisX3[x], self.IrisX4[x]]])
                self.training_label.extend( [self.IrisY[x]])
        ##################### Testing #############################
        for i in range(30, 150, 50):
            for x in range(i, i + 20):
                self.testing_data.extend([[self.IrisX1[x], self.IrisX2[x], self.IrisX3[x], self.IrisX4[x]]])
                self.testing_label.extend([self.IrisY[x]])


        ##############################################################
        self.training_data = np.asarray( self.training_data)
        self.training_label = np.asarray(self.training_label)
        self.testing_data = np.asarray(self.testing_data)
        self.testing_label = np.asarray(self.testing_label)

    def TrainingPhase(self, bias, learn_rate, epochs, hdn_num, nern_lyers, activ_fun):
        self.__init__()
        ################# Enter Data ########################3
        self.bias = bias
        self.learnRate =learn_rate
        self.numOfEpochs = epochs
        self.Num_Of_Hdn_Lyrs = hdn_num
        self.Num_Of_Nern_Ech_Lyr = [int(x) for x in list(nern_lyers.split(","))]
        self.Activation_is_Sigmoid = 1 if (activ_fun == 1) else 0 ## 1 --> Sigmoid, 2-->Hyperbolic
        self.FeaturesSplit()

        self.backprba_algo =backPrpagation(self.training_data, self.training_label,self.bias, self.Num_Of_Hdn_Lyrs,
                                           self.Num_Of_Nern_Ech_Lyr, self.numOfEpochs, self.learnRate,
                                           self.Activation_is_Sigmoid)
        self.weights = self.backprba_algo.backProbagationAlgo()
        return

    def TestingPhase(self):
        Accuracy = 0
        conf_matrix = np.zeros([3, 3], dtype='int32')
        for i in range(0, len(self.testing_data)):
            output = self.backprba_algo.predict(self.testing_data[i])
            if( np.array_equal(np.array(output),np.array(self.testing_label[i]))):
                idx = np.where(output==1)
                Accuracy += 1
                conf_matrix[idx,idx] +=1
            else:
                idx1 = np.where(self.testing_label[i] == 1)
                idx2 = np.where(output == 1)
                conf_matrix[idx1, idx2] += 1
        print("Acurracy: " + str((Accuracy/60.0)*100) +"% ")
        print("Confusion Matrix: " + str(conf_matrix))
        return
    



