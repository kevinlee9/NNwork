# coding: utf-8
from __future__ import print_function

import os
import pickle
import argparse

from rnn import *
import time
import matplotlib.pyplot as plt

path = os.path.dirname(os.path.realpath(__file__))


# padding new rows
def padding_data(x_data, target_size):
    b = np.zeros((target_size, x_data.shape[1]))
    b[:x_data.shape[0], :] = x_data
    return b


# transform from 62*n*5 to n*310
def reshape_data(x_data):
    n = x_data.shape[1]
    return np.asarray([x_data[:, i, :].reshape(-1) for i in range(n)])


def load_npz(file_name):
    zip_data = np.load(file_name)
    file_names = zip_data.keys()
    # return np.vstack([reshape_data(zip_data[file_name]) for file_name in file_names])
    # reshape, padding, then return a tensor 15*270*310
    data = np.asarray([padding_data(reshape_data(zip_data[file_name]), params.padding_size) for file_name in file_names])
    lengths = np.asarray([reshape_data(zip_data[file_name]).shape[0] for file_name in file_names])
    lbls = np.load("data/label.npy")

    out = {
        "X_train": data[0:9, :, :],
        "X_test": data[9:15, :, :],
        "lengths_train": lengths[0:9],
        "lengths_test": lengths[9:15],
        "y_train": lbls[0:9],
        "y_test": lbls[9:15]}
    return out


# split video into small pieces, eg: 30
def load_npz_pcs(file_name, window_size=30):
    zip_data = np.load(file_name)
    file_names = zip_data.keys()

    lbls = np.load("data/label.npy")
    out = {}

    def split_data(dtype="train"):
        if dtype == "train":
            datas = [reshape_data(zip_data[file_name]) for file_name in file_names][0:9]
            y = lbls[0:9]
        elif dtype == "test":
            datas = [reshape_data(zip_data[file_name]) for file_name in file_names][9:15]
            y = lbls[9:15]
        else:
            raise Exception()
        out_data = None
        out_lbl = []
        for j in range(len(datas)):
            data = datas[j]
            print(j)
            for i in range(data.shape[0]-window_size):
                if out_data is not None:
                    out_data = np.concatenate((out_data, data[i:i+window_size, :][None]))
                else:
                    out_data = data[i:i+window_size, :][None]
                out_lbl.append(y[j])
        lengths = [window_size] * len(out_lbl)
        return out_data, np.asarray(lengths), np.asarray(out_lbl)

    out["X_train"], out["lengths_train"], out["y_train"] = split_data("train")
    out["X_test"], out["lengths_test"], out["y_test"] = split_data("test")

    return out


def load_npz_all():
    if os.path.exists(path + "/data/all.pkl"):
        out = pickle.load(path + "/data/all.pkl")
    else:
        out = {}
        data1 = load_npz_pcs(path + "/data/01.npz")
        data2 = load_npz_pcs(path + "/data/02.npz")
        data3 = load_npz_pcs(path + "/data/03.npz")
        out["X_train"] = np.vstack((data1["X_train"], data2["X_train"], data3["X_train"]))
        out["X_test"] = np.vstack((data1["X_test"], data2["X_test"], data3["X_test"]))
        out["y_train"] = np.vstack((data1["y_train"], data2["y_train"], data3["y_train"]))
        out["y_test"] = np.vstack((data1["y_test"], data2["y_test"], data3["y_test"]))
        out["lengths_train"] = np.vstack((data1["lengths_train"], data2["lengths_train"], data3["lengths_train"]))
        out["lengths_test"] = np.vstack((data1["lengths_test"], data2["lengths_test"], data3["lengths_test"]))

        with open(path + "/data/all.pkl", "wb") as f:
            pickle.dump(out, f)
    return out


def load_bypkl(file_name):
    return pickle.load(open(file_name, "rb"))


def load_lbs(file_name):
    return np.load(file_name)


def sample_batch(X_train, lengths, y_train, batch_size):
    N, _, _ = X_train.shape
    ind_N = np.random.choice(N, batch_size, replace=False)
    X_batch = X_train[ind_N]
    lengths_batch = lengths[ind_N]
    y_batch = y_train[ind_N]
    return X_batch, lengths_batch, y_batch


def training(args):
    mode = "fine"
    load_bycached = True
    person_number = args.person_number
    params.batch_size = args.batch_size
    params.num_layers = args.layer_number
    params.time_step = args.time_step
    # person_number = "2"  # option: 0, 1, 2, 3. 0 is all
    export_name = "{}_{}_{}_{}_{}".format(args.person_number, args.batch_size, args.epoch_number, args.layer_size, args.time_step)
    print("export name {}".format(export_name))
    if mode == "raw":
        data1 = load_npz(path + "/data/0{}.npz".format(person_number))
        params.batch_size = 9
    else:
        if not load_bycached:
            data1 = load_npz_pcs(path + "/data/0{}.npz".format(person_number))
        else:
            # data1 = load_bypkl(path + "/data/data{}.pkl".format(person_number))
            data1 = load_bypkl(path + "/data/data{}_{}.pkl".format(person_number, args.time_step))

    model = RNN(10)
    op = tf.train.AdamOptimizer(learning_rate=params.learning_rate, beta1=params.beta1)
    train_step = op.minimize(model.loss)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []
    max_acc = 0
    start_time = time.time()
    for i in range(params.max_steps):
        X_batch, lengths_batch, y_batch = sample_batch(data1["X_train"], data1["lengths_train"], data1["y_train"], params.batch_size)
        _, train_loss, train_acc = sess.run([train_step, model.loss, model.accuracy],
                           feed_dict={model.X: X_batch,
                                      model.y: y_batch,
                                      model.lengths: lengths_batch})

        if i%10 == 0:
            test_loss, test_acc = sess.run([model.loss, model.accuracy],
                                           feed_dict={model.X: data1["X_test"],
                                                      model.y: data1["y_test"],
                                                      model.lengths: data1["lengths_test"]})
            print("Step #" + str(i))
            print("Train loss = {}, Accuracy = {}".format(train_loss, train_acc))
            print("Test loss = {}, Accuracy = {}".format(test_loss, test_acc))
            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            test_loss_list.append(test_loss)
            test_acc_list.append(test_acc)
            if test_acc > max_acc:
                max_acc = test_acc

        if i% 100 == 0:
            print("Epoch {}".format(i * params.batch_size / data1["X_train"].shape[0]))

    end_time = time.time()
    print("training time: {}".format(end_time - start_time))
    print("max accuracy: {}".format(max_acc))


    with open("summary.txt", "wa") as f:
        f.write(export_name + "," + str(max_acc) + "\n")

    #### plot
    np.asarray(train_loss_list).dump("results/train_loss_list{}.np".format(export_name))
    np.asarray(train_acc_list).dump("results/train_acc_list{}.np".format(export_name))
    np.asarray(test_acc_list).dump("results/test_acc_list{}.np".format(export_name))
    np.asarray(test_loss_list).dump("results/test_loss_list{}.np".format(export_name))

    # Matlotlib code to plot the loss and accuracies
    eval_indices = range(0, params.max_steps, 10)
    # Plot loss over time
    plt.plot(eval_indices, train_loss_list, 'k-', label="Train Set Loss")
    plt.plot(eval_indices, test_loss_list, 'r--', label="Test Set Loss")
    plt.title('Softmax Loss per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Softmax Loss')
    plt.legend(loc='lower right')
    plt.savefig("results/loss{}.pdf".format(export_name))

    # Plot train and test accuracy
    plt.plot(eval_indices, train_acc_list, 'k-', label='Train Set Accuracy')
    plt.plot(eval_indices, test_acc_list, 'r--', label='Test Set Accuracy')
    plt.title('Train and Test Accuracy')
    plt.xlabel('Generation')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.savefig("results/accuracy{}.pdf".format(export_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--person_number", type=str, default="1")


