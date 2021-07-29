import numpy as np

class AdalineAlgo:
    def __init__(self):
        self.weights =[]
    def AdalineAlgoTrain(self,featureX,featureY,classlabel,eta,epochs,bias, threshold):
        c1, c2 = classlabel[0], classlabel[len(featureX)-1]
        d = [1 if x ==c1 else 0 for x in classlabel]
        m = len(featureX)
        print('c1 = ', c1 , ' --- c2 = ', c2)
        #print(featureX,featureY,classlabel,eta,m,bias)
        weight = [np.random.rand(1)[0], np.random.rand(1)[0], np.random.rand(1)[0]]
        for epoch in range(epochs):
            error = 0
            for i in range(m):
                x = np.array([featureX[i], featureY[i]])
                y = self.activation_fun(self.net_input_fun(x, weight, bias))
                e = d[i] - y
                weight[0] = weight[0] + eta * e
                weight[1:] = weight[1:] + eta * e * x.T
                error += e**2
            weight[0] *= bias
            self.weights = weight
            MSE = error/(2*m)
            if MSE < threshold:
                break
        print('w1: '+str(self.weights[1]),'w2: '+str(self.weights[2]),'b: '+str(self.weights[0]))
        return self.weights[1], self.weights[2], self.weights[0]

    def predict(self,x1,x2):
        x =np.array( [x1, x2])
        y = self.activation_fun(self.net_input_fun(x,self.weights, 1))
        return np.where(y >= 0, 1, 0)
    def net_input_fun(self,X,weight,bias):
        """Calculate net input"""
        output = np.dot(X, weight[1:]) + weight[0] *bias
        return output
    def activation_fun(self,input):
        """Calculate net input"""
        output = input
        return output