from uwnet import *

def conv_net():
    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 1, LRELU, 1),
            make_maxpool_layer(32, 32, 8, 3, 2),
            make_convolutional_layer(16, 16, 8, 16, 3, 1, LRELU, 1),
            make_maxpool_layer(16, 16, 16, 3, 2),
            make_convolutional_layer(8, 8, 16, 32, 3, 1, LRELU, 1),
            make_maxpool_layer(8, 8, 32, 3, 2),
            make_convolutional_layer(4, 4, 32, 64, 3, 1, LRELU, 1),
            make_maxpool_layer(4, 4, 64, 3, 2),
            make_connected_layer(256, 10, SOFTMAX)]
    return make_net(l)

print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test  = load_image_classification_data("cifar/cifar.test",  "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 128
iters = 5000
rate = .01
momentum = .9
decay = .005

m = conv_net()
print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
train_image_classifier(m, train, batch, 2500, rate*.1, momentum, decay)
train_image_classifier(m, train, batch, 2500, rate*.01, momentum, decay)
# for lr in [rate / 10**i for i in range(4)]:
#     print(f'\nlr = {lr}')
#     train_image_classifier(m, train, batch, 1000, lr, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))



# Without batchnorm, the conv_net gets:
# training accuracy: %f 0.6413999795913696
# test accuracy:     %f 0.6011999845504761

# With batchnorm, the conv_net gets:
# training accuracy: %f 0.7133399844169617
# test accuracy:     %f 0.6802999973297119
# We can see the include of batchnorm results in obviously better performance.

# The best results I can get:
# training accuracy: %f 0.7497199773788452
# test accuracy:     %f 0.7024000287055969
# Training: (as shown above)
# 5000 iters lr = 0.01 -> 2500 iters lr = 0.001 -> 2500 iters lr = 0.0001
