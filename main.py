import csv
import pandas as pd
import cv2
import tensorflow as tf
from functools import partial
import numpy as np

from net.cls import Autoencoder_Classification_Net




class Sigmoid_Xent_with_Logit(tf.keras.losses.Loss):

    def call(self, y_true, y_pred):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred)
        loss = tf.reduce_sum(loss, axis = [1,2,3])
        return tf.reduce_mean(loss)


class AttrDict(dict):

    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__

def define_config(dataset_size):
    config = AttrDict()
    config.shuffle_buffer = 2000
    config.batch_size = 32
    config.base_lr = 1e-3
    config.log_dir = "./tf_log/"
    config.model_path = './model/titanic_cls.ckpt'
    config.size_of_ds = dataset_size
    config.steps_per_epoch = (config.size_of_ds//config.batch_size) * 10
    

    return config

def define_model_config(num_class):
    config = AttrDict()
    config.mlp_dim = 16
    config.layer_num = 8
    config.out_dim = num_class
    return config


def ds_preprocess(item_x, item_y, num_class):
    x = (item_x/255)*2 - 1 # to -1~1
    recon_y = item_x/255 # 0~1
    y = item_y
    
    one_hot_y = tf.one_hot(y, num_class)
    return {"image":x}, {"cls":one_hot_y, "recon":recon_y}


if __name__=="__main__":

    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.set_visible_devices(gpus[0], 'GPU')
    if gpus:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # tf.data.experimental.enable_debug_mode()
    train_pd = pd.read_csv("./data/train.csv")
    # print(train_pd.head())

    train_array = train_pd.to_numpy()
    test_array = pd.read_csv("./data/test.csv").to_numpy()
    # print("train_array", train_array.shape)

    

    train_x = train_array[:,1:].reshape([-1,28,28,1])
    train_y = train_array[:,0]

    train_x.astype(np.float32)
    train_y.astype(np.int32)

    test_x = test_array.reshape([-1,28,28,1])
    test_x.astype(np.float32)



    temp_pair = list(zip(list(train_x), list(train_y)))
    np.random.shuffle(temp_pair)

    train_x, train_y = zip(*temp_pair[1000:])
    val_x, val_y = zip(*temp_pair[:1000])
    train_x, train_y = np.array(train_x), np.array(train_y) 
    val_x, val_y = np.array(val_x), np.array(val_y)

    # release train_array
    train_array = 0
    test_array = 0

    print("val_x:",val_x.shape)




    '''
    get config and necessary info
    '''
    # print("train_x.shape:", train_x.shape)
    # print("train_y.shape:", train_y)
    train_y_list = train_y.flatten('F').tolist()
    num_class = len(list(set(train_y_list)))
    ds_train_size = len(train_y)
    # print("num_class:",type(num_class))
    
    # cv2.imwrite("test.png", train_x[10000])
    print("ds_train_size:",ds_train_size)
    config = define_config(ds_train_size)
    model_config = define_model_config(num_class)
    
    
    '''
    make dataset
    '''
    partial_ds_preprocess = partial(ds_preprocess, num_class = num_class)
    ds_train_x = tf.data.Dataset.from_tensor_slices(train_x)
    ds_train_y = tf.data.Dataset.from_tensor_slices(train_y)
    ds_train = tf.data.Dataset.zip((ds_train_x, ds_train_y))
    ds_train = ds_train.map(partial_ds_preprocess,  tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.repeat()
    ds_train = ds_train.shuffle(config.shuffle_buffer)
    ds_train = ds_train.batch(config.batch_size, drop_remainder=False)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    ds_val_x = tf.data.Dataset.from_tensor_slices(val_x)
    ds_val_y = tf.data.Dataset.from_tensor_slices(val_y)
    ds_val = tf.data.Dataset.zip((ds_val_x, ds_val_y))
    ds_val = ds_val.map(partial_ds_preprocess,  tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.repeat()
    ds_val = ds_val.batch(config.batch_size, drop_remainder=False)
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

    one_train_data = next(ds_train.as_numpy_iterator())[0]








    # for one_item in ds_train.take(100):
    #     print(one_item[0][0])

    ae_cls_net = Autoencoder_Classification_Net(model_config.mlp_dim, model_config.layer_num, model_config.out_dim, "mnist_AE")

    model_input = tf.keras.Input(shape=one_train_data["image"].shape[1:],name="image",dtype=tf.float32)

    model_result = ae_cls_net(model_input)

    model = tf.keras.Model(inputs = {"image":model_input},outputs = {"cls":model_result[0], "recon":model_result[1]}, name = "cls_model")



    # for one_item in ds_train.take(1):
    #     result = model(one_item[0])
    #     print(result)





    # define callback 
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir= config.log_dir, histogram_freq=1, update_freq = 100)
    save_model_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=config.model_path,
        save_weights_only= True,
        verbose=1)

    callback_list = [tensorboard_callback,save_model_callback]

    optimizer = tf.keras.optimizers.Adam(learning_rate = config.base_lr)

    sigmoid_xent = Sigmoid_Xent_with_Logit()

    model.compile(
        optimizer=optimizer, 
        loss={"cls":tf.keras.losses.CategoricalCrossentropy(from_logits=True),"recon":sigmoid_xent},
        metrics = {"cls":tf.keras.metrics.CategoricalAccuracy(), "recon":tf.keras.metrics.BinaryCrossentropy(from_logits = True)}
        )

    print(model.summary())
    print(ae_cls_net.summary())

    


    hist = model.fit(ds_train,
                epochs=2, 
                steps_per_epoch=config.steps_per_epoch,
                # steps_per_epoch=1,
                validation_data = ds_val,
                validation_steps=32,callbacks = callback_list).history

    

    # last_predict = tf.argmax(tf.nn.softmax(model(test_x)), axis = -1).numpy()

    # print(last_predict)
    # print(last_predict.shape) # (418,)


    test_result = model(test_x[:10])
    test_x_0 = (test_x[0]+1)/2*255
    test_recon = test_result["recon"][0]*255
    inference_img = np.concatenate([test_recon,test_x_0], axis = 1)
    inference_img = inference_img.reshape([28,-1,1])
    # print(inference_img.shape)

    cv2.imwrite("inference_img.png", inference_img)
