import numpy as np

class PerceptronAlgo:
    def __init__(self):
        pass
    def perceptronAlgoTrain(self,featureX,featureY,classlabel,eta,m,bias):
        i = 0
        error = 1
        c1 = classlabel[0]
        c2 = classlabel[len(featureX)-1]
        print ('c1 = ', c1 , ' --- c2 = ', c2)
        #print(featureX,featureY,classlabel,eta,m,bias)
        epoch = 0
        weight = [np.random.rand(1)[0], np.random.rand(1)[0], np.random.rand(1)[0]]
        while  epoch < m:
            #if error == 0 or epoch == m:
                #break
            for i in range(0,len(featureX)):
                x = [featureX[i],featureY[i]]
                d = classlabel[i]
                v = self.net_input(x,weight,bias)
                y = self.predict(v)
                if d == c1:
                    d = 1
                else:
                    d = -1
                error = d - y
                weight[0] = weight[0] + eta * error
                weight[1] = weight[1] + eta * error * x[0]
                weight[2] = weight[2] + eta * error * x[1]
            epoch = epoch + 1
            w1 = weight[1]
            w2 = weight[2]
            b = weight[0] * bias
        print('w1: '+str(w1),'w2: '+str(w2),'b: '+str(b))
        return (w1,w2,b)
    def perceptronAlgoTest(self,featureX,featureY,w,bias):
        x = [featureX,featureY]
        v = self.net_inputTest(x,w,bias)
        y = self.predict(v)
        return y
    def net_input(self,X,weight,bias):
        """Calculate net input"""
        v = bias * weight[0] + weight[1]*X[0]+weight[2]*X[1]
        return v
    def net_inputTest(self,X,weight,bias):
        """Calculate net input"""
        v = bias + weight[0]*X[0]+weight[1]*X[1]
        return v

    def predict(self,v):
        d = 0
        if(v>=0.0):
            d = 1
        else:
            d = -1
        return d