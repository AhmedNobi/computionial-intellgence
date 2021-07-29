from Draw_Data import ReadData
from NN_Algo import AdalineAlgo
import numpy as np

class Logic_management:
    def __init__(self):
        self.X1_training = []
        self.X2_training = []
        self.X1_testing = []
        self.X2_testing = []
        self.classlable_training = []
        self.classlable_testing = []

        self.rd = ReadData()
        self.Ada = AdalineAlgo()

        self.c1 = 0
        self.c2 = 0
        self.feature1 = 0
        self.feature2 = 0
        self.w1 = 0
        self.w2 = 0
        self.b = 0
        self.threshold = 0
################## part1: Data Visualizetions for all features###############################
    def possibleCombinations(self):
        rd = ReadData()
        rd.draw_iris_dataset()
#############################################################################################

########################## part2: Train Data #################################################
    def FeaturesSplit(self, chosenFeatures, choosenClasses):
        # initilize X1 and X2
        featureX, featureY = self.initilizeData(chosenFeatures)
        x1_max = max(featureX)
        x1_min = min(featureX)
        x2_min = min(featureY)
        x2_max = max(featureY)
        self.c1 = int(choosenClasses[1])
        self.c2 = int(choosenClasses[6])
        ####
        if self.c1 == 1:
            ##################### Training ###############################
            self.X1_training = featureX[0:30]
            self.X2_training = featureY[0:30]
            self.classlable_training = [1 for i in range(0, 30)]
            ##################### Testing #############################
            self.X1_testing = featureX[30:50]
            self.X2_testing = featureY[30:50]
            self.classlable_testing = [1 for i in range(0, 20)]
            ######################################################
            if self.c2 == 2:
                ##################### Training ###############################
                self.X1_training.extend(featureX[50:80])
                self.X2_training.extend(featureY[50:80])
                self.classlable_training.extend([2 for i in range(0, 30)])
                #################### Testing ################################
                self.X1_testing.extend(featureX[80:100])
                self.X2_testing.extend(featureY[80:100])
                self.classlable_testing.extend([2 for i in range(0, 20)])
                ######################################################3
            else:
                ##################### Training ###############################
                self.X1_training.extend(featureX[100:130])
                self.X2_training.extend(featureY[100:130])
                self.classlable_training.extend([3 for i in range(0, 30)])
                ##################### Testing ################################
                self.X1_testing.extend(featureX[130:150])
                self.X2_testing.extend(featureY[130:150])
                self.classlable_testing.extend([3 for i in range(0, 20)])
                #######################################################
        elif self.c1 == 2:
            ##################### Training ###############################
            self.X1_training = featureX[50:80]
            self.X2_training = featureY[50:80]
            self.classlable_training = [2 for i in range(0, 30)]
            self.X1_training.extend(featureX[100:130])
            self.X2_training.extend(featureY[100:130])
            self.classlable_training.extend([3 for i in range(0, 30)])
            ##################### Testing ##########################################
            self.X1_testing = featureX[80:100]
            self.X2_testing = featureY[80:100]
            self.classlable_testing = [2 for i in range(0, 20)]
            self.X1_testing.extend(featureX[130:150])
            self.X2_testing.extend(featureY[130:150])
            self.classlable_testing.extend([3 for i in range(0, 20)])
            ##############################################################
        self.X1_training = ((np.asarray(self.X1_training)) - x1_min) / (x1_max- x1_min)
        self.X2_training = ((np.asarray(self.X2_training)) - x2_min) / (x2_max- x2_min)
        self.X1_testing = ((np.asarray(self.X1_testing)) - x1_min) / (x1_max- x1_min)
        self.X2_testing = ((np.asarray(self.X2_testing)) - x2_min) / (x2_max- x2_min)

    def TrainingPhase(self, bias, eta, epochsNo, chosenFeatures, choosenClasses, threshold):
        self.__init__()
        self.threshold = threshold
        self.FeaturesSplit(chosenFeatures, choosenClasses)
        #print(self.X1_training, self.X2_training, self.classlable_training, eta, epochsNo, bias)
        w1, w2, b = self.Ada.AdalineAlgoTrain( self.X1_training, self.X2_training, self.classlable_training, eta, epochsNo, bias, self.threshold)
        self.w1 = w1
        self.w2 = w2
        self.b = b
###################################################################################



    def TestingPhase(self):
        Accuracy = 0 #count correct points prediction
        conf = np.zeros([2, 2], dtype='int32')
        print('-------------------------------------------------------------------------')
        for i in range(0,len(self.X1_testing)):
            y = self.Ada.predict(self.X1_testing[i],self.X2_testing[i])
            if (y == 1 and self.c1 == self.classlable_testing[i]):
                conf[0, 0] = conf[0, 0] + 1
                Accuracy+= 1
            elif (y == 0 and self.classlable_testing[i] == self.c2):
                conf[1,1] = conf[1,1] + 1
                Accuracy += 1
            elif (y == 1 and self.classlable_testing[i] != self.c1):
                conf[0, 1] = conf[0, 1] + 1
            elif (y == 0 and self.classlable_testing[i] != self.c2):
                conf[1, 0] = conf[1, 0] + 1

            print('X1: ' + str(self.X1_testing[i]), 'X2: ' + str(self.X2_testing[i]))
            print('Y: ' + str(y))
        print('classes: ' + str(self.classlable_testing))
        print('confusion matrix: ' + str(conf))
        print('Accuracy: ' + str((Accuracy/40)*100) +'%')


    def plotFeatures(self):
        if self.w1 !=0 and self.w2 != 0:
            class_ = [self.c1, self.c2]
            feature_ = [self.feature1, self.feature2]
            self.rd.discriminative_line(self.X1_testing, self.X2_testing, class_, feature_, self.w1, self.w2, self.b)

###################### part: Helper Fun. ####################################
    def initilizeData(self, chosenFeatures):
        self.feature1 = int(chosenFeatures[1])
        self.feature2 = int(chosenFeatures[6])
        self.rd.readData()
        featureX= self.returnFeature(self.feature1)
        featureY = self.returnFeature(self.feature2)
        return featureX, featureY

    def returnFeature(self, index):
        feature = []
        if index == 1:
            feature = self.rd.IrisX1
        elif index == 2:
            feature = self.rd.IrisX2
        elif index == 3:
            feature = self.rd.IrisX3
        else:
            feature = self.rd.IrisX4
        return feature
