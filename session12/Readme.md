Explanation :

    tf.enable_eager_execution()

Eager execution :

Eager execution evaluates without building graphs i.e we can treat tf programs like normal programs
by using print statements. We can use print() statments to debug .

init pytorch() :

The initialisation used in Davidnet which was implemented in pytorch is used here also .Inverse square root of the layers fan in is taken as bound and then generates a random initial weight in range [-bound,bound]


DavidNet class:

David net is constructed with one covolutional block and then three resnet block with two blocks of residual connections and then a global maxpool and then a dense layer.

taking the weights and linear multiplication happens for form logit and then cross entropy loss is calcualted along witht he accuracy .

Dataset is loaded in to xtrain and ytrain and then preprocessing is done (normalisation and padding)

Then learning rate scheduler is used and then optimiser is set with momentum and then data augumentation is used by flipping right left and then with a random crop 

    lr_schedule = lambda t: np.interp([t], [0, (EPOCHS+1)//5, EPOCHS], [0, LEARNING_RATE, 0])[0]
    lr_func = lambda: lr_schedule(global_step/batches_per_epoch)/BATCH_SIZE

loss is scaled up by a factor of batch size .So each gradient would be bathsize*x larger so learning rate should be scaled down by that factor . Also t represents the globalstep per batch/epoch which is interpolated with epochs and learning rate 

Then for each epoch loss and accuracy are calculated 

    opt = tf.train.MomentumOptimizer(lr_func, momentum=MOMENTUM, use_nesterov=True)

Momentnum is varied inversely as learning rate .if learning rate is lower and then higher momentum is highered and then lowered


from_tensor_slices():

Creates a Dataset whose elements are slices of xtest and ytest .We can get slices of array in the form of objects by using from_tensor_slices()

prefetch() :
it is used to prefetch images in the bracket (1) is mentioned .it means batch size is 1

    for g, v in zip(grads, var):
        g += v * WEIGHT_DECAY * BATCH_SIZE
        opt.apply_gradients(zip(grads, var), global_step=global_step)

This is a custom weight decay written as momentumWoptimizer was giving a error and it performs two vector to scalar multiplication 

    train_loss += loss.numpy()    
    test_loss += loss.numpy()

Both train and test loss are seperatly added because to capture total train and test loss for all the epochs 