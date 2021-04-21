import tensorflow as tf

# extend tf.keras' layer API
# add an additional method get_equation_string(), that returns a string mathematical representation of the activation function
# make sure the variable name starts with 'S_' and ends in '_E', so that the str.replace() function won't replace similar-looking variable names

class Identity(tf.keras.layers.Layer):
    def __init__(self): super(Identity,self).__init__()
    def call(self,inputs): return inputs
    def get_equation_string(self,variable_name): return f'S_{variable_name}_E'

class Reciprocal(tf.keras.layers.Layer):
    def __init__(self): super(Reciprocal,self).__init__()
    def call(self,inputs): return tf.math.reciprocal(inputs)
    def get_equation_string(self,variable_name): return f'(1/S_{variable_name}_E)'

class Sine(tf.keras.layers.Layer):
    def __init__(self): super(Sine,self).__init__()
    def call(self,inputs): return tf.sin(inputs)

class Cosine(tf.keras.layers.Layer):
    def __init__(self): super(Cosine,self).__init__()
    def call(self,inputs): return tf.cos(inputs)

class Tangent(tf.keras.layers.Layer):
    def __init__(self): super(Tangent,self).__init__()
    def call(self,inputs): return tf.tan(inputs)

class Log(tf.keras.layers.Layer):
    def __init__(self): super(Log,self).__init__()
    def call(self,inputs): return tf.math.log(inputs)

##class Log2(tf.keras.layers.Layer):
##    def __init__(self): super(Log2,self).__init__()
##    def call(self,inputs): return tf.math.divide( tf.math.log(inputs) , tf.math.log(tf.constant(2, dtype=numerator.dtype)) )
##
##class Log10(tf.keras.layers.Layer):
##    def __init__(self): super(Log10,self).__init__()
##    def call(self,inputs): return tf.math.divide( tf.math.log(inputs) , tf.math.log(tf.constant(10, dtype=numerator.dtype)) )

class Square(tf.keras.layers.Layer):
    def __init__(self): super(Square,self).__init__()
    def call(self,inputs): return tf.square(inputs)
    def get_equation_string(self,variable_name): return f'(S_{variable_name}_E**2)'

class SquareRoot(tf.keras.layers.Layer):
    def __init__(self): super(SquareRoot,self).__init__()
    def call(self,inputs): return tf.sqrt(inputs)

##class Cube(tf.keras.layers.Layer):
##    def __init__(self): super(Cube,self).__init__()
##    def call(self,inputs): return tf.math.pow(inputs,3)
##
##class CubeRoot(tf.keras.layers.Layer):
##    def __init__(self): super(CubeRoot,self).__init__()
##    def call(self,inputs): return tf.math.pow(inputs,1/3)
