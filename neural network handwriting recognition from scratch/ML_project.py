# -*- c oding: utf-8 -*-
import os
import random as rn
from pathlib import Path
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except:
    print("Seaborn is uninstalled; plot styles will differ from report.")
    sns = None
import pandas as pd
from matplotlib.gridspec import GridSpec

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD


#########################################################
# Flags                                                 #
#########################################################

save_plots = True                    # Choose whether to save plots
show_plots = True                    # Choose whether to display them

#########################################################
# Data management functions                             #
#########################################################

def load(filename, load_fn, working_dir=None):
    """
    Call a loading function on a filename.
    If no file is found, return None.
    :param name: string, name of file including extension
    :load_fn: function to call on the path, which returns the object
    :return: data: MNIST data set.
    """
    if working_dir is None: working_dir = Path.cwd()
    else: working_dir = Path(working_dir)
    get_file = str(working_dir / filename)
        # `str` required for Python <= 3.5
    try:
        return load_fn(str(get_file))
    except OSError:
        return None

def split_data():
    train_set, train_label = np.zeros((0, 28*28)), np.zeros((0, 10))
    test_set, test_label = np.zeros((0, 28*28)), np.zeros((0, 10))

    for i in range(10):
        train_set = np.vstack((train_set, ((np.array(MNIST["train"+str(i)])[:])/255)))
        test_set = np.vstack((test_set, ((np.array(MNIST["test"+str(i)])[:])/255)))

        one_hot = np.zeros(10)
        one_hot[i] = 1

        train_label = np.vstack((train_label, np.tile(one_hot, (len(MNIST["train"+str(i)]), 1))))
        test_label = np.vstack((test_label, np.tile(one_hot, (len(MNIST["test"+str(i)]), 1))))

    train_set, train_label, test_set, test_label = train_set.T, train_label.T, test_set.T, test_label.T
    return train_set, train_label, test_set, test_label

class MNIST:
    """
    x_train: concatenated training data sets
             Dimensions are (N,M), where N: feature size, M: no. of samples
    x_test:  concatenated test data sets
             Dimensions are (N',M'), where N': feature size, M': no. of samples
    y_train: 1-D vector of length M
    y_test:  1-D vector of length M'
    y_mat_train:  1-hot matrix of training labels. NxM
    y_mat_test:   1-hot matrix of test labels. NxM'
    """
    labels = "0 1 2 3 4 5 6 7 8 9".split(' ')
    data_shape = (28**2,)
    nlabels = len(labels)

    def __init__(self, filename):
        self.data = load(filename, scipy.io.loadmat)
        # Data values are between 0-255; normalize them to 0-1
        for k, v in self.data.items():
            if 'train' in k or 'test' in k:
                self.data[k] = v / 255.0
        # Create concatenated data sets
        self.x_train = np.vstack([D for D in self.train_data]).T
        self.x_test = np.vstack([D for D in self.test_data]).T
        self.y_train = np.concatenate(
            [[i]*D.shape[0] for i, D in enumerate(self.train_data)] )
        self.y_test = np.concatenate(
            [[i]*D.shape[0] for i, D in enumerate(self.test_data)] )
        k = len(self.labels)
        M, Mp = len(self.y_train), len(self.y_test)
        self.y_mat_train = np.zeros((k, M), dtype=np.float)
        self.y_mat_test = np.zeros((k, Mp), dtype=np.float)
            # Must be a real type for gradient descent
        for i, y in enumerate(self.y_train):
            self.y_mat_train[y, i] = 1
        for i, y in enumerate(self.y_test):
            self.y_mat_test[y, i] = 1

    def __len__(self):
        return len(self.y_train)

    @property
    def train_data(self):
        """Returns a generator expression for the training data.
           Guaranteed to be in the order 0,1,...,9.
        """
        return (self.data['train{}'.format(i)] for i in range(10))
    @property
    def test_data(self):
        """Returns a generator expression for the training data."""
        return (self.data['test{}'.format(i)] for i in range(10))

#########################################################
# Plotting                                              #
#########################################################

if sns:
    sns.set()  # Use Seaborn defaults
    sns.set_palette('colorblind')
    nodrop_colour, drop_colour = sns.color_palette()[:2]
else:
    nodrop_colour, drop_colour = 'blue', 'orange'
train_style = {'linestyle': '--', 'marker': 'x', 'markersize': 4}
test_style = {'linestyle': '-', 'marker': 'o', 'markersize': 4}


plot_in = os.getcwd() + '/plots'
os.makedirs(plot_in, exist_ok=True)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['savefig.bbox'] = 'tight'
    # Makes output size unpredictable; for the true perfectionist: 'standard'
plt.rcParams['savefig.pad_inches'] = 0
#plt.rcParams['savefig.transparent'] = True
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.format'] = 'pdf'

# ---------------------------------------------------------------------
# Part 1

def plot_data_samples(MNIST, fig=None):
    if fig is None: fig = plt.fig()
    img_indices = np.random.random_integers(0, 5000, 10)
    ax = fig.subplots(10,10)

    for i in range(10):
        for j in range(10):
            ax[i,j].imshow(MNIST["train"+str(i)][img_indices[j]].reshape((28,28)))
            ax[i,j].axis('off')
    return fig, ax

# ---------------------------------------------------------------------
# Parts 5 & 8

def plot_training_samples(images, true_labels, predictions=None,
                 nrows=4, ncols=5, fig=None):
    if fig is None: fig = plt.figure()
    if predictions is None: predictions = [None]*len(images)
    N = min(len(images), nrows*ncols)
    ncols = min(N, ncols)
    nrows = np.ceil(N/ncols).astype(int)
    images = images[:N]
    true_labels = true_labels[:N]
    predictions = predictions[:N]
    k = 0
    for img, prediction, label in zip(images, predictions, true_labels):
        k += 1
        ax = fig.add_subplot(nrows, ncols, k)
        ax.imshow(img.reshape(28, 28),
                  cmap='gray', interpolation='none')
        ax.set_title(
            "Predicted: {}\nTrue: {}".format(prediction, label))
        ax.axis('off')
    return fig, ax

def plot_learning_curves(epochs, history,
                         fig=None, axes=None, color=None,
                         labels=("Train", "Test"),
                         xlabel='epoch',
                         ylabel_acc='accuracy (%)', ylabel_loss='loss'):
    if fig is None: fig = plt.figure()
    if axes  is None:
        gs = GridSpec(1, 2)
        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[0,1])
    else:
        ax1, ax2 = axes

    ax1.plot(epochs, np.array(history['acc'])*100,  # Plot percentages
             color=color, **train_style,
             linewidth=2, label=labels[0])
    ax1.plot(epochs, np.array(history['val_acc'])*100,
             color=color, **test_style,
             linewidth=2, label=labels[1])
    ax1.set_title("Accuracy")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel_acc)
    ax1.set_ylim(top=100)

    ax2.plot(epochs, history['loss'],
             color=color, **train_style,
             linewidth=2, label=labels[0])
    ax2.plot(epochs, history['val_loss'],
             color=color, **test_style,
             linewidth=2, label=labels[1])
    ax2.set_title("Loss")
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel_loss)

    # Remove any legend and create an updated one
    for legend in fig.legends:
        legend.remove()
    fig.legend(*ax1.get_legend_handles_labels(),
               loc='lower center', bbox_to_anchor=(0.5, 0), ncol=2)

    fig.tight_layout(rect=[0, 0.16, 1, 0.95])

    return fig, (ax1, ax2)

# ---------------------------------------------------------------------
# Parts 6 & 9

def visualize_unit_weights(weights, units, maxcols=4, fig=None, LNN=False):
    """
    weights: Iterable of all weight matrices corresponding to the units.
    units:   Indices of the units for which we want to print the weights.
    maxcols: Number of columns in the output. Default: 4.
    fig:     If provided, axes on drawn on this figure.
    LNN:     Bool. Changes figure title for the LNN.
    """

    if fig is None: fig = plt.figure()
    ncols = min(len(units), maxcols)
    nrows = np.ceil(len(units) / ncols).astype(int)
    gs = GridSpec(nrows, ncols+1,
                  width_ratios= [1]*ncols + [ncols/20])
    cax = fig.add_subplot(gs[:, -1])  # Colorbar axes
    # Ensure all plots use same colour scale
    vmax = max(weights[:,u].max() for u in units)
    vmin = max(weights[:,u].min() for u in units)
    axes = []
    for i, u in enumerate(units):
        w = weights[:, u].reshape(28, 28)
        ax = fig.add_subplot(gs[i + i//ncols])
        if LNN:
            ax.set_title("Weights for {}".format(u))
        else:
            ax.set_title('Weights to unit {}'.format(u+1))
        cbar=(i==len(units)-1)  # Only print colorbar on the last axes
        if sns:
            sns.heatmap(w, vmin=vmin, vmax=vmax, cmap='coolwarm', cbar_ax=cax)
        else:
            print("Seaborn required for weight matrix visualizatons.")
        ax.axis('off')
        axes.append(ax)
    return fig, axes, cax
# ---------------------------------------------------------------------

#########################################################
# Linear Neural Network                                 #
#########################################################

# ---------------------------------------------------------------------
# Part 2

def softmax(o):
    return np.exp(o)/sum(np.exp(o))

def forward(W, x, b):
    B = np.tile(b, x.shape[1])
    if B.shape == (10, 10):
        import pdb; pdb.set_trace()
    o = np.dot(W.T, x) + B
    P = np.zeros((10,x.shape[1]))
    for i in range(x.shape[1]):
        P[:,i] = softmax(o[:,i])
    return P

# ---------------------------------------------------------------------
# Part 3

def Neg_Log_proba(W,x_train,b, y_train):
    return -np.sum(y_train*np.log(forward(W, x_train, b)))

def gradient(X, Y, W, b):
    N = 10
    P = forward(W, X, b)
    dO = P - Y
    dW = np.dot(X, dO.T)
    db = np.sum(dO, 1).reshape(N, 1)
    return dW, db

# ---------------------------------------------------------------------
# Part 4

def gradient_finite_diff(x_train, y_train , W, b, i, j, h=1e-7):
    deltaij = np.zeros(W.shape)
    deltaij[i, j] = h

    cost   = Neg_Log_proba(W,x_train,b, y_train)
    cost_h = Neg_Log_proba(W+deltaij,x_train,b, y_train)

    return (cost_h - cost)/h

# ---------------------------------------------------------------------
# Part 5 - Training

def predict(x, w, b):
    return np.argmax(forward(w, x, b), axis=0)

def accuracy(x, w, b, y):
    corr = 0
    yhat = predict(x, w, b)
    for i in range(y.shape[1]):
        if (y[yhat[i], i] == 1):
            corr += 1
    return corr/float(y.shape[1])

def train_nn(x_train, y_train, x_test, y_test,  max_it = 35):

    c = load("LNN_fit_cache_{}.npz".format(max_it), np.load)
    if c is not None:
        # Return previous fit from disk cache
        return (c['W'], c['b'], c['itera'],
                c['train_accuracy'], c['test_accuracy'],
                c['train_costs'], c['test_costs'])

    itera , train_accuracy, test_accuracy = [], [], []
    train_costs , test_costs = [], []

    init_W = np.zeros((28*28, 10))
    init_b = np.zeros((10, 1))

    W = init_W.copy()
    b = init_b.copy()
    alpha = 0.01
    batch_size = 50

    for t in range(max_it):
        indices = np.random.permutation(x_train.shape[1])
        x_train = x_train[:,indices]
        y_train = y_train[:,indices]

        for it in range(0, x_train.shape[1], batch_size):
            x_train_i = x_train[:,it:it+batch_size]
            y_train_i = y_train[:,it:it+batch_size]

            dW, db =  gradient(x_train_i, y_train_i, W, b)
            W = W  -  alpha/batch_size * dW
            b = b  -  alpha/batch_size * db


        train_accuracy_i = accuracy(x_train, W, b, y_train)
        test_accuracy_i = accuracy(x_test, W, b, y_test)
        train_cost_i = Neg_Log_proba(W,x_train,b, y_train)
        test_cost_i = Neg_Log_proba(W,x_test,b, y_test)

        itera.append(t)
        train_accuracy.append(train_accuracy_i)
        test_accuracy.append(test_accuracy_i)
        train_costs.append(train_cost_i)
        test_costs.append(test_cost_i)

        print("itreration: " + str(t))
        print("Training Performance:   " + str(train_accuracy_i) + "%")
        print("Testing Performance:    " + str(test_accuracy_i) + "%")
        print("Training cost:   " + str(train_cost_i) )
        print("Testing cost:    " + str(test_cost_i) + "\n")

        # Terminate if there is no more gain on the test set
        if len(test_accuracy) > 2:
            Δtrain = (train_accuracy[-1] - train_accuracy[-2])/train_accuracy[-1]
            Δtest = (test_accuracy[-1] - test_accuracy[-2])/test_accuracy[-1]
            if Δtrain < 0.001 and Δtest < 0.001:
                break

    np.savez("LNN_fit_cache_{}.npz".format(max_it), W=W, b=b, itera=itera,
             train_accuracy=train_accuracy, test_accuracy=test_accuracy,
             train_costs=train_costs, test_costs=test_costs)
    return W, b, itera, train_accuracy, test_accuracy, train_costs, test_costs

# ---------------------------------------------------------------------
# Parts 5 & 8 - Correct vs incorrect samples

def split_correct_indices(predictions, true, shuffle=True):
    correct_indices = np.nonzero(predictions == true)[0]
    incorrect_indices = np.nonzero(predictions != true)[0]
    if shuffle:
        np.random.shuffle(correct_indices)
        np.random.shuffle((incorrect_indices))
    return {'correct': correct_indices, 'incorrect': incorrect_indices}
# ---------------------------------------------------------------------


#########################################################
# Nonlinear Neural Network (using Keras)                #
#########################################################

class KerasModel:

    def __init__(self, data_type=MNIST, dropout=False, bsize=50, n_epochs=120):
        self.bsize = bsize
        self.n_epochs = n_epochs
        self.dropout = dropout
        self.fnn = load(self.filename, keras.models.load_model)
        self.history = load(self.hist_filename, np.load)
        if self.fnn is None:
            self.fnn = self.make_KerasNN(data_type, dropout=dropout)

# ---------------------------------------------------------------------
# Part 7

    @staticmethod
    def make_KerasNN(data_type, dropout=False):
        """
        Hard-coded parameters:
            - Hidden layer: 300 units
            - SGD: (lr=0.1, momentum=0.3, decay=0, nesterov=True)
        """
        fnn = Sequential()
        fnn.add(Dense(300, activation='tanh', input_shape=data_type.data_shape))
        if dropout:
            fnn.add(Dropout(0.5))
        fnn.add(Dense(data_type.nlabels, activation='softmax'))

        sgd = SGD(lr=0.1, momentum=0.3, decay=0, nesterov=True)
        fnn.compile(optimizer=sgd, loss='categorical_crossentropy',
                    metrics=['accuracy'])
        return fnn
# ---------------------------------------------------------------------


    @property
    def filename(self):
        return 'fnn-batch-{}-epochs-{}-dropout-{}.h5'\
               .format(self.bsize, self.n_epochs,
                       'yes' if self.dropout else 'no')
    @property
    def epochs(self):
        return np.arange(1, self.n_epochs+1)
    @property
    def hist_filename(self):
        basename = str(Path(self.filename).stem)
        return "{}_history.npz".format(basename)
    @property
    def test_loss(self):
        return self.res[0]
    @property
    def test_acc(self):
        return self.res[1]

# ---------------------------------------------------------------------
# Part 8

    def fit(self, xdata, ydata, validation_data=None):
        if self.history is None:
            self.fnn.fit(xdata, ydata, validation_data=validation_data,
                         batch_size=self.bsize, epochs=self.n_epochs)
            self.history = {k:np.array(v)
                            for k, v in self.fnn.history.history.items()}
            self.fnn.save(self.filename)
            np.savez(self.hist_filename, **self.fnn.history.history)
                # Save history so we can plot performance curves

    def evaluate(self, xdata, ydata):
        return self.fnn.evaluate(xdata, ydata, batch_size=self.bsize)
# ---------------------------------------------------------------------

#########################################################
# Main Script                                           #
#########################################################

if __name__ == "__main__":

    data = MNIST('mnist_all.mat')                      # Load data

    # Set seeds and force single threading for reproducible results------------
    np.random.seed(5314)
    rn.seed(53140)
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    tf.set_random_seed(5314)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    keras.backend.set_session(sess)

    # Part 1 - Data samples -----------------------------------------------
    fig1 = plt.figure(figsize=(6,6))
    fig1, ax = plot_data_samples(data.data, fig=fig1)
    if show_plots: fig1.show()
    if save_plots: fig1.savefig(plot_in + '/fig1.pdf')

    # Part 2 - LNN implementation ------------------------------------------
    #forward(W, data.x_train, b)

    # Part 3 - LNN gradient implementation ---------------------------------
    #gradient(train_set, train_label, W, b)

    # Part 4 - LNN gradient check
    samples_i = [288, 600, 100]
    samples_j = [8, 4, 1]
    for i, j in zip(samples_i, samples_j):
        W = np.random.uniform(0, 1, (28*28, 10))
        b = np.random.uniform(0,1,(10, 1))
        gf = gradient_finite_diff(data.x_train, data.y_mat_train, W, b, i, j)
        g = gradient(data.x_train, data.y_mat_train, W, b)[0][i, j]
        print('W[{:d}, {:d}]'.format(i, j))
        print('--------------------------------------')
        print('grad_dWij: {:.7f}\nfinite-difference approximation: {:.7f}'
              .format(g, gf))
        print('absolute difference: {:.7f}\n'.format(abs(g - gf)))

    # Part 5 - LNN training -----------------------------------------------

    # Training
    W, b, epochs, train_accuracy, test_accuracy, train_costs, test_costs \
        = train_nn(data.x_train, data.y_mat_train,
                   data.x_test, data.y_mat_test, max_it = 35)
    # Collect training history into similar structure to Keras history
    train_history = {'acc': train_accuracy, 'val_acc': test_accuracy,
                     'loss': train_costs,   'val_loss': test_costs}

    # Performance
    fig3 = plt.figure(figsize=(6,3.3))
    fig3, axes = plot_learning_curves(epochs, train_history,
                                      fig=fig3,
                                      color= nodrop_colour,
                                      labels=('Train', 'Test'))
    if show_plots: fig3.show()
    if save_plots: fig3.savefig(plot_in + '/fig3.pdf')

    # Correctly vs incorrectly classified figures
    with plt.style.context({'axes.titlesize': 10}):
        predictions = predict(data.x_test, W, b)
        indices = split_correct_indices(predictions, data.y_test)
        indices['correct'] = indices['correct'][:20]
        indices['incorrect'] = indices['incorrect'][:10]
        fig2a = plt.figure(figsize=(6,6))
        plot_training_samples(data.x_test.T[indices['correct']],
                     data.y_test[indices['correct']],
                     predictions[indices['correct']], fig=fig2a)
        fig2a.subplots_adjust(hspace=0.6, wspace=0.25)
        fig2b = plt.figure(figsize=(6,3))
        plot_training_samples(data.x_test.T[indices['incorrect']],
                     data.y_test[indices['incorrect']],
                     predictions[indices['incorrect']], fig=fig2b)
        fig2b.subplots_adjust(hspace=0.6, wspace=0.25)

    if show_plots:
        fig2a.show()
        fig2b.show()
    if save_plots:
        fig2a.savefig(plot_in + '/fig2a.pdf')
        fig2b.savefig(plot_in + '/fig2b.pdf')

    # Part 6 - LNN weight visualization -----------------------------------
    fig4 = plt.figure(figsize=(6, 6))
    fig4, axes, cax = visualize_unit_weights(W, range(10), maxcols=3, LNN=True, fig=fig4)
    if show_plots: fig4.show()
    if save_plots: fig4.savefig(plot_in + '/fig4.pdf')

    # Part 7 - Keras model ------------------------------------------------

    model1 = KerasModel(dropout=False, n_epochs=30)
    model2 = KerasModel(dropout=True, n_epochs=30)

    # Part 8 - Keras training ---------------------------------------------
    train_data = (data.x_train.T, data.y_mat_train.T)
    test_data = (data.x_test.T, data.y_mat_test.T)
    model1.fit(*train_data, test_data)
    model2.fit(*train_data, test_data)

    # Plot performance
    fig5 = plt.figure(figsize=(6,3.3))
    fig5, axes = plot_learning_curves(model1.epochs, model1.history,
                                      fig=fig5,
                                      color = nodrop_colour,
                                      labels = ("Train (no dropout)",
                                                "Test (no dropout)"),
                                      ylabel_loss = 'cross-entropy loss')
    plot_learning_curves(model2.epochs, model2.history,
                         fig=fig5, axes=axes,
                         color = drop_colour,
                         labels = ("Train (with dropout)",
                                   "Test (with dropout)"))
    fig5.suptitle('Neural network with hidden layer')

    if show_plots: fig5.show()
    if save_plots: fig5.savefig(plot_in + '/fig5.pdf')

    # Results summary
    xtest = data.x_test.T
    ytest = data.y_mat_test.T
    res = pd.DataFrame(
        {('Nonlinear Neural Network', 'No dropout') : model1.evaluate(xtest, ytest),
         ('Nonlinear Neural Network', 'Dropout')    : model2.evaluate(xtest, ytest)},
       index=['Test set loss', 'Test set accuracy (%)'])
    for i in range(2):
        res.iloc[i,:] = res.iloc[i,:].map(lambda x: '{0:.3}'.format(x))
    res.loc['Test set accuracy (%)'] *= 100   # Convert to percentage
    with open(plot_in + "/result_table.tex", 'w') as f:
        f.write(res.to_latex())

    # Correctly vs incorrectly classified figures
    with plt.style.context({'axes.titlesize': 10}):
        predictions = model1.fnn.predict_classes(data.x_test.T)
        indices = split_correct_indices(predictions, data.y_test)
        indices['correct'] = indices['correct'][:20]
        indices['incorrect'] = indices['incorrect'][:10]
        fig6 = plt.figure(figsize=(6,6))
        plot_training_samples(data.x_test.T[indices['correct']],
                     data.y_test[indices['correct']],
                     predictions[indices['correct']], fig=fig6)
        fig6.subplots_adjust(hspace=0.6, wspace=0.25)
        fig7 = plt.figure(figsize=(6,3))
        plot_training_samples(data.x_test.T[indices['incorrect']],
                     data.y_test[indices['incorrect']],
                     predictions[indices['incorrect']], fig=fig7)
        fig7.subplots_adjust(hspace=0.6, wspace=0.25)

    if show_plots:
        fig6.show()
        fig7.show()
    if save_plots:
        fig6.savefig(plot_in + '/fig6.pdf')
        fig7.savefig(plot_in + '/fig7.pdf')

    # -------------------------------------------------------------------
    # Part 9 - Weight Visualization

    fig8 = plt.figure(figsize=(6.5, 3))
    units = [212, 230]
    weights0 = model1.fnn.layers[0].get_weights()[0]
    fig8, axes, cax = visualize_unit_weights(weights0, units=units, fig=fig8);

    if show_plots: fig8.show()
    if save_plots: fig8.savefig(plot_in + '/fig8.pdf')

    # --------------------------------------------------------------------

    if show_plots:
        # Prevent immediate termination of script
        input("Press any key to exit.\n")
