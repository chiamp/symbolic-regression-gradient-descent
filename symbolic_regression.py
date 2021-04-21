import tensorflow as tf
from tensorflow.keras.layers import Input,Dense,Multiply,Concatenate,BatchNormalization
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop,Adam
from tensorflow.keras.regularizers import L1

import custom_layers
from custom_layers import *

import numpy as np
from sklearn.preprocessing import StandardScaler

from sympy import simplify,expand,nsimplify

import time


def mse(x,y,model): return np.mean( ( y.reshape(-1,1) - model([x[:,i] for i in range(x.shape[1])]) )**2 )


def initialize_model(config):
    # initialize neural network model to fit the data to
    # model will have access to all activation functions in config
    # model will also have Dense (linear combination) and Multiply layers, that will combine all inputs and activated inputs together
    # all of these outputs are used as inputs for the next operation block of the neural network
    
    input_layers = [Input(1) for _ in range(config['network']['input_dimension'])]
    
    previous_layers = [layer for layer in input_layers] # to hold the input layers / previous operator layers
    activation_function_layers = [] # to hold the activation layers
    activation_batch_norm_layers = [] # to hold the batch norm layers after activation function layers
    operator_layers = [] # to hold the operator layers
    final_batch_norm_layers = [] # to hold the batch norm layers after activation function and operator layers
    for _ in range(config['network']['num_layers']): # iterate for the number of operation blocks in the neural network

        # feed inputs / previous layer outputs into activation functions
        for previous_layer in previous_layers:
            for activation_function_layer in config['network']['activation_functions']:
                activation_function_layers.append( activation_function_layer()(previous_layer) )

        # batch normalize the activation function outputs
        activation_batch_norm_layers = [BatchNormalization()(layer) for layer in activation_function_layers]

        # keep a copy of the outputs of the batch normalized activation function outputs, and then perform operations on a separate copy
        final_batch_norm_layers = [layer for layer in activation_batch_norm_layers]

        # concatenate batch normalized activation function outputs and feed into Dense layer (for addition/subtraction operations)
        concat_layer = Concatenate()(activation_batch_norm_layers)
        operator_layers.append( Dense(1,activation=None,kernel_regularizer=L1(config['optimizer']['regularization_constant']),bias_regularizer=L1(config['optimizer']['regularization_constant']))(concat_layer) )

        # feed each batch normalized activation function output into a separate Dense neuron (of size 1)
        # the weight will act as a "selector", determining whether to use this particular activation output value, and to what extent, in the upcoming Multiply layer (for multiplication/division operations)
        selection_layer = [Dense(1, activation=None,kernel_regularizer=L1(config['optimizer']['regularization_constant']),bias_regularizer=L1(config['optimizer']['regularization_constant']))(layer) for layer in activation_batch_norm_layers]
        operator_layers.append( Multiply()(selection_layer) )

        # batch normalize operator outputs, and then concatenate with the batch normalized activation function outputs
        final_batch_norm_layers.extend([BatchNormalization()(layer) for layer in operator_layers])
        
        previous_layers = final_batch_norm_layers
        activation_function_layers = []
        activation_batch_norm_layers = []
        operator_layers = []
        final_batch_norm_layers = []

    # concatenate previous layer outputs and feed them into a single Dense neuron (of size 1) as the output of the entire neural network
    concat_layer = Concatenate()(previous_layers)
    output_layer = Dense(1,activation=None,kernel_regularizer=L1(config['optimizer']['regularization_constant']),bias_regularizer=L1(config['optimizer']['regularization_constant']))(concat_layer)

    model = Model(input_layers,output_layer)
    optimizer = Adam(learning_rate=config['optimizer']['learning_rate'])
    model.compile(optimizer=optimizer,loss='mse')

    return model

def extract_equation(model):
    # extract an equation string from the model, equivalent to the model's forward prop / inference

    # start with the last layer, build a mathematical expression based on previous layer inputs
    # then recursively expand the previous layer inputs into mathematical expressions based on the subsequent previous layer inputs
    # continue until you reach the input layer(s)

    # layer inputs are represented by their layer name attribute, starting with 'S_' and ending with '_E'
    
    #### NOTE: if new activation functions / operators are used, they must be included in the if statement code block below
    
    equation_string = '' # the final equation

    queue = [model.layers[-1]]
    while len(queue) > 0:
        layer = queue.pop(0)
        previous_layers = layer.inbound_nodes[0].inbound_layers # get the previous layers connected to this layer
        # if the previous layer is a Concatenate layer, get the subsequent previous layer (NOTE: this assumes there can't be consecutive Concatenate layers)
        if type(previous_layers) == tf.python.keras.layers.merge.Concatenate: previous_layers = previous_layers.inbound_nodes[0].inbound_layers
        # if there is only one previous layer (as opposed to multiple previous layers connected to our current layer), the return type will be the layer itself, not a list, so we must place it in a list
        if type(previous_layers) != list: previous_layers = [previous_layers]

        current_layer_equation = [] # the equation of the current layer

        ###### if statement code block containing all possible activation function / operator layers ######
        if type(layer) == tf.python.keras.layers.core.Dense:
            weight_matrix = layer.weights[0].numpy()
            bias = layer.weights[1].numpy()[0]
            assert len(previous_layers) == layer.weights[0].numpy().shape[0] # make sure the number of previous layers connected to this layer, matches the number of weights we have

            for i in range(len(previous_layers)):
                current_layer_equation.append(f'({weight_matrix[i,0]}*S_{previous_layers[i].name}_E)')
                queue.append(previous_layers[i])
            current_layer_equation.append(f'{bias}')
            current_layer_equation = '+'.join(current_layer_equation)

        elif type(layer) == tf.python.keras.layers.merge.Multiply:
            for i in range(len(previous_layers)):
                current_layer_equation.append(f'S_{previous_layers[i].name}_E')
                queue.append(previous_layers[i])
            current_layer_equation = '*'.join(current_layer_equation)

        elif type(layer) == tf.python.keras.layers.normalization_v2.BatchNormalization:
            # previous layer should only have single output
            assert len(previous_layers) == 1
            assert layer.moving_mean.numpy().size == 1
            current_layer_equation = f'( ( ( S_{previous_layers[0].name}_E - {layer.moving_mean.numpy()[0]} ) / ({layer.moving_variance.numpy()[0]} + {layer.epsilon})**(1/2) ) * {layer.gamma.numpy()[0]} + {layer.beta.numpy()[0]} )'
            queue.append(previous_layers[0])

        elif type(layer) == tf.python.keras.engine.input_layer.InputLayer:
            # assume all InputLayers have a name attribute value of 'input_{int}'
            # rename the variable into 'X{int}'
            current_layer_equation = f"X{layer.name.split('_')[-1]}"
            # nothing is before an input layer so nothing to add to the queue

        else: # must be an activation function from custom_layers module
            assert len(previous_layers) == 1 # all activation functions take a singular input
            current_layer_equation = layer.get_equation_string(previous_layers[0].name) # get the mathematical expression string equivalent of this layer's activation function
            queue.append(previous_layers[0])

##        else: raise Exception(f'Layer type error: {type(layer)}')

        ###################################################################################################

        if equation_string == '': equation_string = current_layer_equation # start with the last layer's equation, and expand from there
        else: equation_string = equation_string.replace(f'S_{layer.name}_E',f'({current_layer_equation})') # replace all variables with the current layer's name with the math equation of current_layer_equation

        equation_string = str(expand(equation_string)) # expand the equation string after every loop iteration to simplify (it would take a much longer time if we expanded only after the entire equation string is built)

    return equation_string

def unnormalize_equation(equation_string,x_scaler,y_scaler): # unnormalize the equation_string, given the mean and stdev's of data x and target y
    unnormalized_equation = equation_string
    
    for i in range(x_scaler.mean_.size):
        variable = f'X{i+1}' # input variables are 1-indexed (assumption based on how tf.keras names their layers)
        unnormalized_equation = unnormalized_equation.replace(variable,f'( ( {variable}-{x_scaler.mean_[i]} ) / {x_scaler.scale_[i]} )') # replace input variables with the unnormalized form

    unnormalized_equation = f'{y_scaler.scale_[0]} * ({unnormalized_equation}) + {y_scaler.mean_[0]}' # factor in unnormalizing the target

    return unnormalized_equation

def run(x_original,y_original,config):
    # z-normalize data
    x_scaler = StandardScaler()
    x = x_scaler.fit_transform(x_original)
    y_scaler = StandardScaler()
    y = y_scaler.fit_transform(y_original.reshape(-1,1))

    num_validation_samples = round(x.shape[0]*config['training']['validation_percentage'])
    val_x = x[-num_validation_samples:,:]
    x = x[:-num_validation_samples,:]
    val_y = y[-num_validation_samples:]
    y = y[:-num_validation_samples]

    # initialize model
    model = initialize_model(config)
    print(f'Number of trainable parameters: { sum([variable_matrix.numpy().size for variable_matrix in model.trainable_variables]) }')

    # training loop
    current_learning_rate = config['optimizer']['learning_rate']
    
    train_loss = mse(x,y,model)
    train_loss_values = [train_loss]
    average_train_loss = sum(train_loss_values)/len(train_loss_values)
    
    validation_loss = mse(val_x,val_y,model)
    validation_loss_values = [validation_loss]
    average_validation_loss = sum(validation_loss_values)/len(validation_loss_values)
    
    print(f"\nMSE train loss\tMSE past {config['training']['last_n_loss_values']} average train loss\tMSE validation loss\tMSE past {config['training']['last_n_loss_values']} average validation loss\tCurrent learning rate")
    print(f'{train_loss}\t{average_train_loss}\t{validation_loss}\t{average_validation_loss}\t{current_learning_rate}')
    
    start_time = time.time()
    while (train_loss > config['training']['loss_cutoff']) and (current_learning_rate > config['training']['learning_rate_cutoff']):
        # fit model
        model.fit([x[:,0],x[:,1]],y,batch_size=x.shape[0],epochs=config['training']['epoch_interval'],verbose=0)
        train_loss = mse(x,y,model)
        validation_loss = mse(val_x,val_y,model)

        # update train_loss_values and validation_loss_values
        if len(train_loss_values) >= config['training']['last_n_loss_values']: train_loss_values.pop(0)
        train_loss_values.append(train_loss)
        if len(validation_loss_values) >= config['training']['last_n_loss_values']: validation_loss_values.pop(0)
        validation_loss_values.append(validation_loss)

        # calculate new average loss values
        average_train_loss = sum(train_loss_values)/len(train_loss_values)
        average_validation_loss = sum(validation_loss_values)/len(validation_loss_values)
        
        print(f'{train_loss}\t{average_train_loss}\t{validation_loss}\t{average_validation_loss}\t{current_learning_rate}')
        
        # decrease learning rate if train_loss is more than the past n average_train_loss
        if (train_loss > average_train_loss) or (validation_loss > average_validation_loss):
            current_learning_rate *= 0.1
            model.compile(optimizer=Adam(learning_rate=current_learning_rate),loss='mse')

    print(f'Training converged in {(time.time()-start_time)/60} minutes')

    start_time = time.time()
    equation_string = extract_equation(model)
    unnormalized_equation = unnormalize_equation(equation_string,x_scaler,y_scaler)
    print(f'Equation extracted and unnormalized in {time.time()-start_time} seconds')

    start_time = time.time()
    expanded_equation = expand(unnormalized_equation)
    print(f'Equation expanded in {time.time()-start_time} seconds')

    start_time = time.time()
    simplified_equations = [str( nsimplify(expanded_equation,tolerance=10**-tolerance_value) ) for tolerance_value in range(config['max_digits_tolerance'])]
    print(f'Equation simplified in {time.time()-start_time} seconds')

    print('\nTesting...')
    best_simplified_equation_index = test(x_original,y_original,x_scaler,y_scaler,model,unnormalized_equation,simplified_equations,config['print_test_predictions'])

    print(f'\nUnsimplified equation: {unnormalized_equation}')
    print(f'\nExpanded equation: {expanded_equation}')
    print(f'\nSimplified equation (best tolerance level: {10**-best_simplified_equation_index}): {simplified_equations[best_simplified_equation_index]}\n')

    return unnormalized_equation,simplified_equations[best_simplified_equation_index]

def test(x,y,x_scaler,y_scaler,model,unnormalized_equation,simplified_equations,print_predictions=False):
    # test the explicit unnormalized_equation and simplified_equation output and check how close the predicted output is with the true value
    # also compare results to predicted model output; should be the same as unnormalized_equation output
    
    pred = model( [ ( (x[:,i] - x_scaler.mean_[i]) / x_scaler.scale_[i] ) for i in range(x.shape[1]) ] )
    pred = ( (pred * y_scaler.scale_[0]) + y_scaler.mean_[0] ).numpy().reshape(-1)

    pred_outputs = []
    pred_errors = []
    unnormalized_equation_outputs = []
    unnormalized_equation_errors = []
    simplified_equation_outputs = [[].copy() for _ in range(len(simplified_equations))]
    simplified_equation_errors = [[].copy() for _ in range(len(simplified_equations))]
    for i in range(x.shape[0]):

        unnormalized_output = unnormalized_equation
        for feature_index in range(x.shape[1]):
            unnormalized_output = unnormalized_output.replace(f'X{feature_index+1}',str(x[i,feature_index]))

        simplified_outputs = simplified_equations.copy()
        for j,simplified_equation in enumerate(simplified_outputs):
            for feature_index in range(x.shape[1]):
                simplified_outputs[j] = simplified_outputs[j].replace(f'X{feature_index+1}',str(x[i,feature_index]))

        pred_outputs.append( pred[i] )
        pred_errors.append( abs(y[i]-pred_outputs[-1]) )
        unnormalized_equation_outputs.append( eval(unnormalized_output) )
        unnormalized_equation_errors.append( abs(y[i]-unnormalized_equation_outputs[-1]) )
        for j,simplified_output in enumerate(simplified_outputs):
            simplified_equation_outputs[j].append( eval(simplified_output) )
            simplified_equation_errors[j].append( abs(y[i]-simplified_equation_outputs[j][-1]) )
    
    summed_simplified_equation_errors = [ sum(simplified_equation_error_list) for simplified_equation_error_list in simplified_equation_errors ]
    best_simplified_equation_index = int(np.argmin(summed_simplified_equation_errors))
    
    if print_predictions:
        print('\nReal values','model_prediction','unnormalized_equation','simplified_equation','model_error','unnormalized_equation_error','simplified_equation_error',sep='\t')
        for i in range(x.shape[0]):
            print( y[i] , pred_outputs[i] , unnormalized_equation_outputs[i] , simplified_equation_outputs[best_simplified_equation_index][i] , pred_errors[i] , unnormalized_equation_errors[i] , simplified_equation_errors[best_simplified_equation_index][i] , sep='\t' )

    print(f'Average Model Prediction Absolute Error: {sum(pred_errors)/x.shape[0]}')
    print(f'Average Unnormalized Equation Absolute Error: {sum(unnormalized_equation_errors)/x.shape[0]}')
    print(f'Average Simplified Equation Absolute Error: {sum(simplified_equation_errors[best_simplified_equation_index])/x.shape[0]}')

    return best_simplified_equation_index

if __name__ == '__main__':
    config = {'network':{'input_dimension':2, # number of features in your data
                         'activation_functions':[Identity,Square], #,Reciprocal
                         'num_layers':1, # number of operation blocks in your neural network
                         },
              'optimizer':{'regularization_constant':0*1e-3, # regularization parameter for all weight values
                           'learning_rate':1e-3, # learning rate of the model
                           },
              'training':{'validation_percentage':0.15, # percentage of dataset to be partitioned for validation testing
                          'last_n_loss_values':10, # take the average of the last n loss values; if the training loss is more than the average, lower the learning rate
                          'loss_cutoff':1e-14, # stop training the model once loss value is below this cutoff
                          'learning_rate_cutoff':1e-12, # stop training the model if learning rate decreases to below this value
                          'epoch_interval':1000, # train for epoch_interval epochs, and then print MSE loss
                          },
              'max_digits_tolerance':12, # try rounding the extracted equation from the nearest integer (0 digits), to max_digits_tolerance digits, and keep the simplified equation that performs the best
              'print_test_predictions':True, # boolean to denote whether to print test prediction values
              }
    
    x = np.random.rand(100,config['network']['input_dimension']) * 10
    y = (0.5*x[:,0]**2 + 5*x[:,1]) + 4

    unnormalized_equation,simplified_equation = run(x,y,config)
    

### TODO ###
### TEST REMOVING BIAS IN SELECTION LAYER
### TEST ADDING MINMAXNORM CONSTRANT FUNCTION TO WEIGHTS IN SELECTION LAYER
### TEST A CUSTOM CONSTRAINT FUNCTION THAT ONLY FLIPS WEIGHT VALUES BETWEEN EITHER 0 OR 1
    ### AND/OR FIND A LAYER OR MAKE A CUSTOM LAYER THAT SELECTS CERTAIN OUTPUTS TO GO THROUGH, AND ZERO THE REST (SOME KIND OF RELU/HARDMAX THING)
### TEST (RANDOM, NOT SYSTEMATIC) GRID SEARCH FOR HYPERPARAMETERS, USE COARSE-TO-FINE SAMPLING SCHEME
    ### SOME HYPER PARAMETERS LIKE LEARNING RATES, MAY DO BETTER IF YOU SAMPLE ON THE LOG SCALE, THAN UNIFORMLY (1E-4 VS 1)
### FIGURE OUT WHY BATCH NORMALIZATION LAYERS RESULT IN WORSE PERFORMANCE
### TEST ADDING MOMENTUM=0.9 HYPER PARAMETER VALUE TO RMSPROP OPTIMIZER
### CUSTOM LOSS FUNCTION, THAT PENALIZES BASED ON THE NUMBER OF NON-ZERO WEIGHTS (SOME KIND OF SUMMATION OF PENALTY TERM MULTIPLIED BY INDICATOR VARIABLES THAT ARE 1 IF WEIGHT IS NON-ZERO, OTHERWISE ZERO)
