import pandas as pd
import logging
from pathlib import Path
import numpy as np
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from wide_resnet import WideResNet
from utils import load_data
from keras.preprocessing.image import ImageDataGenerator
from mixup_generator import MixupGenerator
from matplotlib import pyplot as plt


logging.basicConfig(level=logging.DEBUG)


class Schedule:
    def __init__(self, nb_epochs, initial_lr):
        self.epochs = nb_epochs
        self.initial_lr = initial_lr

    def __call__(self, epoch_idx):
        if epoch_idx < self.epochs * 0.25:
            return self.initial_lr
        elif epoch_idx < self.epochs * 0.50:
            return self.initial_lr * 0.2
        elif epoch_idx < self.epochs * 0.75:
            return self.initial_lr * 0.04
        return self.initial_lr * 0.008


def main():
    input_path = 'data/imdb_db.mat'
    batch_size = 32
    nb_epochs = 2
    lr = 0.1
    depth = 16
    k = 8
    validation_split = 0.1
    output_path = "data"


    logging.debug("Loading data...")
    image, gender, age, _, image_size, _ = load_data(input_path)
    X_data = image
    y_data_g = np_utils.to_categorical(gender, 2)
    y_data_a = np_utils.to_categorical(age, 101)

    model = WideResNet(image_size, depth=depth, k=k)()
    opt = Adam(lr=lr)
    model.compile(optimizer=opt, loss=["categorical_crossentropy", "categorical_crossentropy"],
                  metrics=['accuracy'])

    logging.debug("Model summary...")
    model.count_params()
    model.summary()

    callbacks = [LearningRateScheduler(schedule=Schedule(nb_epochs, lr)),
                 ModelCheckpoint(str(output_path) + "/weights.{epoch:02d}-{val_loss:.2f}.hdf5",
                                 monitor="val_loss",
                                 verbose=1,
                                 save_best_only=True,
                                 mode="auto")
                 ]

    logging.debug("Running training...")

    data_num = len(X_data)
    indexes = np.arange(data_num)
    np.random.shuffle(indexes)
    X_data = X_data[indexes]
    y_data_g = y_data_g[indexes]
    y_data_a = y_data_a[indexes]
    train_num = int(data_num * (1 - validation_split))
    X_train = X_data[:train_num]
    X_test = X_data[train_num:]
    y_train_g = y_data_g[:train_num]
    y_test_g = y_data_g[train_num:]
    y_train_a = y_data_a[:train_num]
    y_test_a = y_data_a[train_num:]

    datagen = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        preprocessing_function=get_random_eraser(v_l=0, v_h=255))
    training_generator = MixupGenerator(X_train, [y_train_g, y_train_a], batch_size=batch_size, alpha=0.2,
                                        datagen=datagen)()
    hist = model.fit(training_generator,
                               steps_per_epoch=train_num // batch_size,
                               validation_data=(
                                   X_test, [y_test_g, y_test_a]),
                               epochs=nb_epochs, verbose=1,
                               callbacks=callbacks)
    print(hist.history.keys())

    #logging.debug("Saving history...")
    #pd.DataFrame(hist.history).to_hdf(output_path.joinpath("history_{}_{}.h5".format(depth, k)), "history")


    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    #plt.plot(hist.history['pred_gender_accuracy'])
    #plt.plot(hist.history['val_pred_gender_accuracy'])
    #plt.title('model accuracy')
    #plt.ylabel('accuracy')
    #plt.xlabel('epoch')
    #plt.legend(['train', 'val'], loc='upper left')
    #plt.show()

    plt.savefig('foo.png')
    plt.savefig('foo.pdf')

    


def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255):
    def eraser(input_img):
        img_h, img_w, _ = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        c = np.random.uniform(v_l, v_h)
        input_img[top:top + h, left:left + w, :] = c

        return input_img

    return eraser


if __name__ == '__main__':
    main()
