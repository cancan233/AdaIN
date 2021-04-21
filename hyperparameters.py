"""
Hyperparameters used in models
"""

"""
Number of training epochs
"""
num_epochs = 20

"""
training rate
"""
learning_rate = 1e-4

"""
parameter decides the transfer level, 0 to 1, where more close to 1 means deeper style transfer
"""
alpha = 1

"""
parameter to avoid divide by 0
"""
epsilon = 1e-5

"""
batch size
"""
batch_size = 8

"""
style loss weight
"""
style_lambda = 10


"""
image size
"""
img_size = 256

"""
input image size
"""
input_shape = (None, None, 3)
