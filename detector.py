import keras
from keras import Sequential
from keras.callbacks import LearningRateScheduler
from keras.datasets import mnist
import os
import keras.backend.tensorflow_backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
from keras.layers import Dense, BatchNormalization, Dropout
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

def load_feature(attack_name):


    train_SF_path = ''
    train_SF_adv_path = ''
    train_NSF_path = ''
    train_NSF_adv_path = ''

    test_SF_path = ''
    test_SF_adv_path = ''
    test_NSF_path = ''
    test_NSF_adv_path = ''


    train_SF_pre = np.load(train_SF_path)
    train_SF_adv_pre = np.load(train_SF_adv_path)
    train_NSF_pre = np.load(train_NSF_path)
    train_NSF_adv_pre = np.load(train_NSF_adv_path)

    test_SF_pre = np.load(test_SF_path)
    test_SF_adv_pre = np.load(test_SF_adv_path)
    test_NSF_pre = np.load(test_NSF_path)
    test_NSF_adv_pre = np.load(test_NSF_adv_path)

    # train datasets
    train_ori_SF_and_NSF = np.concatenate((train_SF_pre, train_SF_adv_pre), axis=1)
    train_ori_label = np.zeros(shape=(len(train_ori_SF_and_NSF), 1))
    train_adv_SF_and_NSF = np.concatenate((train_NSF_pre, train_NSF_adv_pre), axis=1)
    train_adv_label = np.ones(shape=(len(train_adv_SF_and_NSF), 1))
    print(np.shape(train_ori_SF_and_NSF), np.shape(train_adv_label))
    train_SF_and_NSF = np.concatenate((train_ori_SF_and_NSF, train_adv_SF_and_NSF))
    train_SF_and_NSF_label = np.concatenate((train_ori_label, train_adv_label))

    # test datasets
    test_ori_SF_and_NSF = np.concatenate((test_SF_pre, test_SF_adv_pre), axis=1)
    test_ori_label = np.zeros(shape=(len(test_ori_SF_and_NSF), 1))
    test_adv_SF_and_NSF = np.concatenate((test_NSF_pre, test_NSF_adv_pre), axis=1)
    test_adv_label = np.ones(shape=(len(test_adv_SF_and_NSF), 1))
    print(np.shape(test_ori_SF_and_NSF), np.shape(test_ori_label))
    test_SF_and_NSF = np.concatenate((test_ori_SF_and_NSF, test_adv_SF_and_NSF))
    test_SF_and_NSF_label = np.concatenate((test_ori_label, test_adv_label))

    # SF_and_NSF_label = keras.utils.to_categorical(SF_and_NSF_label,num_classes=2)
    print("train:", train_SF_and_NSF_label[0], train_SF_and_NSF_label.shape, train_SF_and_NSF.shape)
    print("test:", test_SF_and_NSF_label[0], test_SF_and_NSF_label.shape, test_SF_and_NSF.shape)
    print("-" * 10, attack_name, "-" * 10)

    train_x = train_SF_and_NSF
    train_y = train_SF_and_NSF_label
    test_x = test_SF_and_NSF
    test_y = test_SF_and_NSF_label


    return train_x,train_y,test_x,test_y




def MLP(dropout_rate=0.25, activation='relu',classes=1):
    start_neurons = 512
    model = Sequential()
    model.add(Dense(start_neurons, input_dim=256, activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    model.add(Dense(start_neurons // 2, activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    model.add(Dense(start_neurons // 4, activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    model.add(Dense(start_neurons // 8, activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate / 2))

    model.add(Dense(classes, activation='sigmoid'))
    return model





def plot_loss_acc(history, fold, base_path,acc,max_epoch):

    history_dict = history.history
    history_dict.keys()
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epoch = range(1, len(loss_values) + 1)
    plt.plot(epoch, loss_values, label='Training loss')
    plt.plot(epoch, val_loss_values, label='Validation loss')
    plt.title("Training and validation loss at acc:%.2f%%" % acc)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    my_x_ticks = np.arange(0, max_epoch+1,1 )
    plt.xticks(my_x_ticks)
    plt.legend()
    plt.savefig(base_path + "_fold_" + str(fold) + "_loss.png")
    plt.close()

    acc_values = history_dict['acc']
    val_acc_values = history_dict['val_acc']
    plt.plot(epoch, acc_values, label='Training acc')
    plt.plot(epoch, val_acc_values, label='Validation acc')
    plt.title("Training and validation accuracy at acc:%.2f%%" % acc)
    plt.xlabel('Epochs')
    my_x_ticks = np.arange(0, max_epoch+1, 1)
    plt.xticks(my_x_ticks)
    plt.ylabel('Acc')
    plt.legend()
    plt.savefig(base_path + "_fold_" + str(fold) + "_acc.png")
    plt.close()
    np.save(base_path + "_fold_" + str(fold) + ".npy", history_dict)


if __name__=="__main__":

    attack_list = ['BIM', 'MIFGSM', 'JSMA', 'CRA', 'AUNA', 'PWA', 'LSA']
    attack_name = attack_list[0]
    detect_name = attack_list[0]
    detector_history_path = ''
    model_save_path = ''
    train_x,train_y,test_x,test_y = load_feature(attack_name)
    print(train_y[0],train_y[10000])
    classes = 1

    index = np.arange(len(train_x[0:10000]))
    np.random.shuffle(index)
    train_x = train_x[index]
    train_y = train_y[index]

    index = np.arange(len(test_x[0:2000]))
    np.random.shuffle(index)
    test_x = test_x[index]
    test_y = test_y[index]

    K.clear_session()
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)

    folds = KFold(n_splits=5, shuffle=True, random_state=2019)

    patience = 10  ## How many steps to stop
    call_ES = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=1,
                                            mode='auto', baseline=None)
    epochs = 25
    batch_size = 256
    cvscores_train = []
    cvscores_test = []
    checkpoint = keras.callbacks.ModelCheckpoint(filepath=model_save_path, monitor='val_acc', verbose=1,
                                                     save_best_only=True, mode='max')
    model = MLP(dropout_rate=0.5, activation='relu')
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # to uap
    sgd = keras.optimizers.SGD(lr=0.0001, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

    model.load_weights(model_save_path)
    scores = model.evaluate(train_x[0:10000], train_y[0:10000], verbose=2)
    print("train_sub_val %s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    cvscores_train.append(scores[1] * 100)
    scores = model.evaluate(test_x[0:2000], test_y[0:2000], verbose=2)
    print("test %s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    exit(0)

    time_start = time.time()
    history = model.fit(train_x[0:10000], train_y[0:10000],
                        # validation_data=[test_x, test_y],
                        validation_split =0.2,
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[call_ES,checkpoint ],
                        shuffle=True,
                        verbose=1)
    time_end = time.time()
    print('totally cost', time_end - time_start)
    time_start = time.time()
    scores = model.evaluate(train_x[0:10000], train_y[0:10000], verbose=2)
    time_end = time.time()
    print('totally cost', time_end - time_start)
    print("train_sub_val %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores_train.append(scores[1] * 100)
    scores = model.evaluate(test_x[0:2000], test_y[0:2000], verbose=2)
    print("test %s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    cvscores_test.append(scores[1] * 100)
    plot_loss_acc(history, 0, detector_history_path, scores[1]*100,max_epoch=epochs)
    print("-" * 10, attack_name, "-" * 10)













