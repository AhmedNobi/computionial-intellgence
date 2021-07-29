import numpy as np

class backPrpagation:
    def __init__(self,train, label, bias, Num_Of_Hdn_Lyrs,
                                           Num_Of_Nern_Ech_Lyr, numOfEpochs,learnRate,
                                           Activation_is_Sigmoid):
        ##################################
        self.bias = bias
        self.learnRate = learnRate
        self.Epochs = numOfEpochs
        self.numOfHidden_Layers = Num_Of_Hdn_Lyrs  ## Hidden Lyers
        self.Num_Of_Nern_Ech_Lyr = Num_Of_Nern_Ech_Lyr
        self.sigmoid_bool = Activation_is_Sigmoid
        ##################################
        ################## data #################

        self.trainig_data =train

        self.trainig_label = label

        self.Neurons_values = [] ##
        self.Neurons_error = []
        self.output_Neurons_error = []

        self.number_of_input_neurons = 4
        self.number_of_output_classes = 3

        self.weights = self.intialize_weights()

        self.a = 1.0
        self.b = 1.0
        for neur in self.Num_Of_Nern_Ech_Lyr:
            self.Neurons_error.append(np.asarray([0.0 for i in range(neur)]))
        self.Neurons_error.append([0.0,0.0,0.0])
        self.Neurons_error = np.asarray(self.Neurons_error)

    def intialize_weights(self):
        weights = []
        num_ofLayers =self.Num_Of_Nern_Ech_Lyr.copy()
        num_ofLayers.insert(0, self.number_of_input_neurons)  ### X1, X2, X3, X4
        num_ofLayers.append(self.number_of_output_classes)  ##3
        layer = 1
        while layer < len(num_ofLayers):
            weights.append(np.array(np.random.rand(num_ofLayers[layer], num_ofLayers[layer - 1]+1)))
            layer += 1
        return weights

    def backProbagationAlgo(self):
        for e in range(self.Epochs):
            for itr in range(len(self.trainig_label)):
                self.forward_step(self.trainig_data[itr], self.trainig_label[itr])
                self.backward_step()
                self.update_step(list(self.trainig_data[itr]))
        print(self.weights)
        return self.weights


    def forward_step(self, input_data,actual_output):
        self.Neurons_values.clear()
        for layer in range(len(self.weights)):
            new_input_data = np.zeros(len(input_data) + 1)
            new_input_data[0] =self.bias
            new_input_data[1:len(new_input_data)] = input_data[:]

            input_data = np.dot(self.weights[layer], new_input_data)

            if self.sigmoid_bool:
                input_data = self.sigmoid(input_data)
            else:
                input_data = self.hyperbolic_tangent_sigmoid(input_data)
            self.Neurons_values.append(input_data)
        self.output_Neurons_error = actual_output - self.Neurons_values[len(self.Neurons_values)-1]

    def backward_step(self):
        last_lyer= len(self.Neurons_error)-1
        self.Neurons_error[last_lyer] = self.output_Neurons_error* self.select_method(self.sigmoid_bool,self.Neurons_values[
               last_lyer], self.a, self.b)

        for lyr in range(self.numOfHidden_Layers-1,-1, -1):
            #print("lyr: "+str(lyr))
            for neuron_idx in range(len(self.Neurons_error[lyr])):
                err = np.asarray(self.Neurons_error[lyr+1]).reshape(len(self.Neurons_error[lyr + 1]), 1)
                #print("err: " + str(err) )
                #print("nu: " + str(self.weights[lyr+1][:,neuron_idx+1]))
                #print("act: "+ str(self.select_method(self.sigmoid_bool,self.Neurons_values[lyr][neuron_idx],1,1)))
                nuron = np.asarray(self.weights[lyr+1][:,neuron_idx+1]).reshape(len(self.weights[lyr+1][:,neuron_idx+1]), 1)

                self.Neurons_error[lyr][neuron_idx] = self.select_method(self.sigmoid_bool,
                             self.Neurons_values[lyr][neuron_idx], self.a, self.b) * np.sum(np.dot(err, nuron.T))

                #print(self.Neurons_error[lyr][neuron_idx])

        #print("########### Errors ######################")
        #print("activation: " + str([self.sigmoid_deffe(x) for x in self.Neurons_values]))
        #print("weights: " + str(self.weights))
        #print(self.Neurons_error)
        #print("#################################")
        return

    def update_step(self, input_data):
        neuronS_Activation_val =[]

        new_input_data = np.zeros(len(input_data) + 1)
        new_input_data[0] = self.bias
        new_input_data[1:len(new_input_data)] = input_data[:]

        neuronS_Activation_val.append(new_input_data)
        for x in self.Neurons_values:
            x = list(x)
            x.insert(0, self.bias)
            neuronS_Activation_val.append(np.asarray(x))
        neuronS_Activation_val = np.asarray(neuronS_Activation_val)

        #print("########### before ######################")
        #print(self.weights)
        #print("#################################")
        for lyr_w in range(len(self.weights)):
            err =np.asarray(self.Neurons_error[lyr_w]).reshape(len(self.Neurons_error[lyr_w]),1)
            nuron =np.asarray(neuronS_Activation_val[lyr_w]).reshape(len(neuronS_Activation_val[lyr_w]),1)
            #print("########### before X1 ######################")
            #print(self.weights[lyr_w])
            #print("#################################")
            self.weights[lyr_w] +=  (self.learnRate * np.dot(err,nuron.T))
            #print("########### before X2######################")
            #print(self.weights[lyr_w])
            #print("#################################")
        #print("########### After ######################")
        #print(self.weights)
        #print("#################################")
        return

    def select_method(self, choice, z, a,b):
        return self.sigmoid_deffe(z) if choice == 1 else self.hyperbolic_diffe(a, b, z)

    def sigmoid_deffe(self, z):
        return z*(1-z)
    def hyperbolic_diffe(self, a, b, z):
        #return float((b/a) * np.dot(np.asarray([a-z]), np.asarray([a+z]).T))
        return 1 - self.hyperbolic_tangent_sigmoid(z) ** 2

    def sigmoid(self, net):
        return 1. / (1. + np.exp(-net))

    def hyperbolic_tangent_sigmoid(self, net):
        return np.tanh(net)
    def add_bias(self,train):
        if (self.bias == 0):
            return np.asarray(np.c_[np.array(np.zeros(train.shape[0])), np.array(train)])
        return  np.asarray(np.c_[np.array(np.ones(train.shape[0])), np.array(train)])
    def predict(self, input_data):
        for layer in range(len(self.weights)):
            new_input_data = np.zeros(len(input_data) + 1)
            new_input_data[0] =self.bias
            new_input_data[1:len(new_input_data)] = input_data[:]

            input_data = np.dot(self.weights[layer], new_input_data)
            if self.sigmoid_bool:
                input_data = self.sigmoid(input_data)
            else:
                input_data = self.hyperbolic_tangent_sigmoid(input_data)
        print(input_data)
        output = np.zeros(input_data.shape)

        output[np.argmax(input_data)] = 1
        print(output)
        return output
