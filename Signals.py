import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Load signal files
labelled_signals = np.loadtxt('Data/labelled_signal_data_epochs.txt')
df = pd.read_csv('Data/dataframe_no_signal.csv')

# Assemble into dataframe
df['signal'] = labelled_signals.tolist()
df['signal'] = df['signal'].apply(np.array)
df['signal'] = df['signal'].apply(lambda x: 1000*x) # Convert from V to mV

# X data will be the unique IDs, they will allow indexing of the signals properly
# y data will be class labels
cls_1_ids = df.loc[df['label']==1, 'id'].unique()
cls_2_ids = df.loc[df['label']==2, 'id'].unique()
cls_3_ids = df.loc[df['label']==3, 'id'].unique()
# ids and labels is order [1, 1, 1, 1, 1, ..., 1, 2, 2, 2, 2, 2, ..., 2]
cls_ids_12 = np.append(cls_1_ids, cls_2_ids)
cls_ids_23 = np.append(cls_2_ids, cls_3_ids)
cls_labels_12 = np.append(np.ones((cls_1_ids.size,)), np.ones((cls_2_ids.size,)) * 2)
cls_labels_23 = np.append(np.ones((cls_2_ids.size,)) * 2, np.ones((cls_3_ids.size,)) * 3)
# Split data
X_train_12, X_test_12, y_train, y_test = train_test_split(cls_ids_12, cls_labels_12, test_size=0.25, random_state=0)
X_train_23, X_test_23, y_train, y_test = train_test_split(cls_ids_23, cls_labels_23, test_size=0.25, random_state=0)


def get_train_signals_2a(cls, channel):
    """
    :param cls: class label (integer 1 or 2)
    :param channel: channel number (integer between 1 and 8)
    :return: N x 256 array of signals of selected class and channel
    """
    cls_signals = df.loc[(df['id'].isin(X_train_12)) & (df['label'] == cls), 'signal'].values.tolist()
    cls_signals = np.array(cls_signals)
    cls_signals = np.reshape(cls_signals, (cls_signals.shape[0] // 8, 8, 256))
    return cls_signals[:, channel-1, :]


def get_train_signals_2b(cls, channel):
    """
    :param cls: class label (integer 2 or 3)
    :param channel: channel number (integer between 1 and 8)
    :return: N x 256 array of signals of selected class and channel
    """
    cls_signals = df.loc[(df['id'].isin(X_train_23)) & (df['label'] == cls), 'signal'].values.tolist()
    cls_signals = np.array(cls_signals)
    cls_signals = np.reshape(cls_signals, (cls_signals.shape[0] // 8, 8, 256))
    return cls_signals[:, channel-1, :]


def test_classifier_2a(classifier_fun):
    """
    Tests accuracy of manual classifier function on test data for class 1 and 2
    :param classifier_fun: function that returns class based on input 8x256 nd-array of a single signal epoch
    :return: Accuracy score
    """
    test_signals = df.loc[(df['id'].isin(X_test_12)), 'signal'].values.tolist()
    test_signals = np.array(test_signals)
    test_signals = np.reshape(test_signals, (test_signals.shape[0] // 8, 8, 256))

    # Get labels from df. Since it's repeated 8 times for each channel, only pull every 8th one
    ys = df.loc[(df['id'].isin(X_test_12)), 'label'].values
    ys = ys[::8]

    classifier_output = [classifier_fun(signal) for signal in test_signals]
    classifier_output = np.array(classifier_output)
    num_correct = np.count_nonzero(classifier_output == ys)
    accuracy = num_correct / ys.size
    print('{} correct out of {} in test dataset for 2a'.format(num_correct, ys.size))
    print('Classifier accuracy on test data: {:.3f}'.format(accuracy))
    return accuracy


def test_classifier_2b(classifier_fun):
    """
    Tests accuracy of manual classifier function on test data for class 2 and 3
    :param classifier_fun: function that returns class based on input 8x256 nd-array of a single signal epoch
    :return: Accuracy score
    """
    test_signals = df.loc[(df['id'].isin(X_test_23)), 'signal'].values.tolist()
    test_signals = np.array(test_signals)
    test_signals = np.reshape(test_signals, (test_signals.shape[0] // 8, 8, 256))

    # Get labels from df. Since it's repeated 8 times for each channel, only pull every 8th one
    ys = df.loc[(df['id'].isin(X_test_23)), 'label'].values
    ys = ys[::8]

    classifier_output = [classifier_fun(signal) for signal in test_signals]
    classifier_output = np.array(classifier_output)
    num_correct = np.count_nonzero(classifier_output == ys)
    accuracy = num_correct / ys.size
    print('{} correct out of {} in test dataset for 2b'.format(num_correct, ys.size))
    print('Classifier accuracy on test data: {:.3f}'.format(accuracy))
    return accuracy


def mean_absolute_value(x):
    x = np.absolute(x)
    total = np.sum(x)
    return total / len(x)


def root_mean_square(x):
    sq = np.square(x)
    total = np.sum(sq)
    mean = total / len(x)
    return np.sqrt(mean)


def variance(x):
    mean = np.sum(x) / len(x)
    total_var = np.sum(np.square(x - mean))
    return total_var / len(x)


def zero_crossings(x):
    x[x != 0]
    return (x[:-1] * x[1:] < 0).sum()


def classifier_fun_2a(x):
    A = root_mean_square(x[7,:])
    B = root_mean_square(x[1,:])
    discriminant = -4*A + 0.18
    if B < discriminant:
        c = 1
    else:
        c = 2
    return c

def classifier_fun_2b(x):
    B = root_mean_square(x[3,:])
    A = root_mean_square(x[0,:])
    discriminant = 1/4 * A + 0.04
    if B < discriminant:
        c = 3
    else:
        c = 2
    return c


cls3_ch1 = get_train_signals_2b(3, 1)
cls3_ch4 = get_train_signals_2b(3, 4)
cls2_ch1 = get_train_signals_2b(2, 1)
cls2_ch4 = get_train_signals_2b(2, 4)

cls3_ch1_rms = np.apply_along_axis(root_mean_square, 1, cls3_ch1)
cls3_ch4_rms = np.apply_along_axis(root_mean_square, 1, cls3_ch4)
cls2_ch1_rms = np.apply_along_axis(root_mean_square, 1, cls2_ch1)
cls2_ch4_rms = np.apply_along_axis(root_mean_square, 1, cls2_ch4)

plt.figure(figsize=(12,8))
plt.scatter(cls3_ch1_rms, cls3_ch4_rms, c='xkcd:green', label='Wrist Flexion Features', s=4)
plt.scatter(cls2_ch1_rms, cls2_ch4_rms, c='xkcd:blue', label='Fist Features', s=4)
x1 = np.linspace(0, 2)
y1 = 1/4 * x1 + 0.04
plt.plot(x1, y1)
plt.xlim((0, 1))
plt.ylim((0, 1))
plt.legend(loc='best')
plt.xlabel('Channel 1, Root Mean Square')
plt.ylabel('Channel 4, Root Mean Square')
plt.show()

#test_classifier_2a(classifier_fun_2a)
test_classifier_2b(classifier_fun_2b)