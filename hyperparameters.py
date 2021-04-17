"""
Hyperparameters used in models
"""

"""
Number of training epochs
"""
num_epochs = 1

"""
training rate
"""
learning_rate = 1e-5


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
batch_size = 1

"""
style loss weight
"""
s_lambda = 1


"""
image size
"""
img_size = 256


"""
Sample size for calculating the mean and standard deviation of the
training data. This many images will be randomly seleted to be read
into memory temporarily.
"""
preprocess_sample_size = 400
