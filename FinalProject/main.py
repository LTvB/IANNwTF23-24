import tensorflow as tf
import numpy as np
from tqdm import tqdm
from tensorflow.keras.datasets import mnist

class FFN:

    def __init__(self, input_shape=(784,), hidden_layers_neuron=[2000, 2000], num_classes=10):
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.hidden_layers_neuron = hidden_layers_neuron
        self.model = self.create_mlp()
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


    def create_mlp(self):
        model = tf.keras.models.Sequential([
                tf.keras.layers.Dense(units=784, input_shape=self.input_shape, activation='relu'),
                tf.keras.layers.Dense(self.hidden_layers_neuron[0], input_shape=self.input_shape, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                Gating_Layer(),
                tf.keras.layers.Dense(self.hidden_layers_neuron[1], activation='relu'),
                tf.keras.layers.Dropout(0.5),
                Gating_Layer(),
                tf.keras.layers.Dense(self.num_classes)
        ])
        return model

    def get_compiled_model(self, metrics):
        compiled_model = self.model
        compiled_model.compile(self.optimizer, self.loss_function, metrics)
        return compiled_model

    def return_hidden_layer(self):
      hidden_layers = self.hidden_layers_neuron
      return hidden_layers

class Gating_Layer(tf.keras.layers.Layer):
  def __init__(self):
    super().__init__()
    self.mask = None

  def call (self, input):
      if self.mask is None:
        return input
      else:
        output = input * self.mask
        return output

class Train:

    def __init__(self, lambda_): 
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.lambda_ = lambda_

    def train(self, model, epochs, train_task, prior_weights=None, fisher_matrix=None, test_tasks=None, gating=None, XdG=False):
        self.optimizer = tf.keras.optimizers.Adam()
        prior_weights = prior_weights
        for epoch in range(epochs):
            epoch_loss_avg = tf.keras.metrics.Mean()
            epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
            for batch in tqdm(train_task):
              X, y = batch
              with tf.GradientTape() as tape:
                  pred = model(X, training=True)
                  loss = self.loss_function(y, pred)
                  #to execute training with EWC
                  if fisher_matrix is not None:
                      aux_loss = self.compute_penalty_loss(model, fisher_matrix, prior_weights)
                      loss += aux_loss
              grads = tape.gradient(loss, model.trainable_variables)
              self.optimizer.apply_gradients(zip(grads, model.trainable_variables))

              epoch_loss_avg.update_state(loss)
              epoch_accuracy.update_state(y, pred)
            if fisher_matrix is not None:
              print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss_avg.result()}, Accuracy: {epoch_accuracy.result()}, aux_loss: {aux_loss}")
            else:
              print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss_avg.result()}, Accuracy: {epoch_accuracy.result()}")

            # reset metric states
            epoch_loss_avg.reset_states()
            epoch_accuracy.reset_states()

            # evaluate with the test set of task after each epoch
            test_accuracies = []
            for i in range(len(test_tasks)):
              if XdG is True:
                gating_for_task = gating[i][0]
                model.layers[3].mask = gating_for_task[0]
                model.layers[6].mask = gating_for_task[1]
              test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
              
              for batch in test_tasks[i]:
                X, y = batch
                preds = model(X)
                test_accuracy.update_state(y, preds)
              print(f"tesk task {i}, Accuracy: {test_accuracy.result()}")
              test_accuracies.append(test_accuracy.result().numpy())
              test_accuracy.reset_states()
            mean = np.mean(np.array(test_accuracies))
            print("Mean:", mean)
        
        return test_accuracies, mean


    def compute_penalty_loss(self, model, fisher_matrix, prior_weights):
        penalty = 0.
        for u, v, w in zip(fisher_matrix, model.weights, prior_weights):
          penalty += tf.math.reduce_sum(u * tf.math.square(v - w))
        return 0.5 * self.lambda_ * penalty


class EWC:

    def __init__(self, prior_model, data_samples, num_ewc_sample=8192):
        self.prior_model = prior_model
        self.prior_weights = prior_model.weights
        self.num_sample = num_ewc_sample
        self.data_samples = data_samples
        self.fisher_matrix = self.compute_fisher()

    def compute_fisher(self):
        weights = self.prior_weights
        fisher_accum = np.array([np.zeros(layer.numpy().shape) for layer in weights],
                           dtype=object
                          )

        for j in tqdm(range(self.num_sample)):
          idx = np.random.randint(self.data_samples.shape[0])
          
          with tf.GradientTape() as tape:
              logits = tf.nn.log_softmax(self.prior_model(np.array([self.data_samples[idx]])))
          grads = tape.gradient(logits, weights)

          for m in range(len(weights)):
              fisher_accum[m] += np.square(grads[m])
        fisher_accum /= self.num_sample

        return fisher_accum

    def get_fisher(self):
        return self.fisher_matrix

class Sy_Int():
  def __init__(self, epochs, param_c=2, param_xi=0.01):
    self.epochs = epochs
    self.param_c = param_c
    self.param_xi = param_xi

  def init_vars(self, model):

    self.small_omega_var = {}
    self.big_omega_var = {}

    for var in model.trainable_variables:
      self.small_omega_var[var.name] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
      self.big_omega_var[var.name] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
    print("Init: small_omega_var, big_omega_var")

  def compute_aux_loss_SI(self, previous_weights): # think I dont need these vars can just use self.
    aux_loss = 0.0
    for var in self.model.trainable_variables:
      aux_loss += tf.reduce_sum(tf.multiply(self.big_omega_var[var.name], tf.square(previous_weights[var.name] - var) ))
    return aux_loss

  def reset_small_omega(self):
    for var in self.model.trainable_variables:
      self.small_omega_var[var.name] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
    print("Small omegas reseted")

  #to execute training with SI
  def train_SI(self, model, train_task, test_tasks, gating=None, XdG=False):
    self.model = model
    self.optimizer = tf.keras.optimizers.Adam()
    optimizer = tf.keras.optimizers.Adam()
    loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    self.previous_weights = {var.name: var.numpy() for var in self.model.trainable_variables}


    for epoch in range(self.epochs):
      epoch_loss_avg = tf.keras.metrics.Mean()
      epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
      
      for batch in tqdm(train_task):
        X, y = batch

        with tf.GradientTape(persistent=True) as tape:
          pred = self.model(X, training=True)
          task_loss = loss_function(y, pred)
          aux_loss = self.compute_aux_loss_SI(self.previous_weights)
          loss = task_loss + self.param_c * aux_loss

        # gradients for training + applying these
        gradients_with_aux = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients_with_aux, self.model.trainable_variables))

        # updating small omegas
        gradients = tape.gradient(task_loss, self.model.trainable_variables)

        for grad, var in zip(gradients, self.model.trainable_variables):
          self.small_omega_var[var.name].assign_add(-optimizer.lr*grad)

        del tape

      epoch_loss_avg.update_state(loss)
      epoch_accuracy.update_state(y, pred)

      print(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss_avg.result()}, Accuracy: {epoch_accuracy.result()}, Aux_loss: {aux_loss}")

      # reset metric states
      epoch_loss_avg.reset_states()
      epoch_accuracy.reset_states()

      # evaluate with the test set of task after each epoch
      test_accuracies = []
      for i in range(len(test_tasks)):
        if XdG is True:
          gating_for_task = gating[i][0]
          model.layers[3].mask = gating_for_task[0]
          model.layers[6].mask = gating_for_task[1]
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        
        for batch in test_tasks[i]:
          X, y = batch
          preds = self.model(X)
          test_accuracy.update_state(y, preds)
        print(f"tesk task {i}, Accuracy: {test_accuracy.result()}")
        test_accuracies.append(test_accuracy.result().numpy())
        test_accuracy.reset_states()
    mean = np.mean(np.array(test_accuracies))
    print("Mean:", mean)

    #update big omega after the task
    for var in self.model.trainable_variables:
      self.big_omega_var[var.name].assign_add(tf.divide(tf.nn.relu(self.small_omega_var[var.name]), \
            (self.param_xi + tf.square(var-self.previous_weights[var.name]))))
    print("Big Omega updated")

    return test_accuracies, mean

def permute_task(train, test):
  train_shape, test_shape = train.shape, test.shape
  idx = np.arange(train.shape[1])
  np.random.shuffle(idx)
  train_permuted, test_permuted = train[:, idx], test[:, idx]
  return (train_permuted, test_permuted)

def make_mnist_permutation(x_train, y_train, x_test, y_test):
  x_train, x_test = permute_task(x_train, x_test)
  y_train, y_test = y_train, y_test
  train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(256)
  test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(1000).batch(256)
  return train, test, x_train

def trainings_loop(epochs, num_tasks, use_EWC=False, lambda_=1000, use_SI=False, param_c=2, param_xi=0.01, use_XdG=False, gating_percentage=0.8, num_ewc_sample=8192):
  epochs = epochs
  num_tasks = num_tasks
  num_ewc_sample = num_ewc_sample

  # load normal MNIST and do preprocessing
  (x_train_MNIST, y_train_MNIST), (x_test_MNIST, y_test_MNIST) = mnist.load_data()
  x_train_MNIST = x_train_MNIST.astype('float32')/255
  x_test_MNIST = x_test_MNIST.astype('float32')/255
  x_train_MNIST = x_train_MNIST.reshape((-1, 784))
  x_test_MNIST = x_test_MNIST.reshape((-1, 784))

  ffn = FFN()
  model_normal = ffn.get_compiled_model(['accuracy'])
  model_normal.summary()

  train_normal = Train(lambda_)
  if use_SI:
    SI = Sy_Int(epochs, param_c=param_c, param_xi=param_xi)
    SI.init_vars(model_normal)

  train_tasks = []
  test_tasks = []
  complete_mean = []
  list_of_accs = []
  gating_masks_per_task = [[] for _ in range(num_tasks)]

  for num in range(num_tasks):
    print('Task number:', num)
    # make MNIST permutation and save test in test_tasks
    train, test, x_train_unbatched = make_mnist_permutation(x_train_MNIST, y_train_MNIST, x_test_MNIST, y_test_MNIST)
    train_tasks.append(train)
    test_tasks.append(test)

    if use_XdG is True:
      layers = ffn.return_hidden_layer()
      gating_task = []
      for n in range(len(layers)):
        gating_layer = np.zeros(layers[n])
        for i in range(gating_layer.shape[0]):
          if np.random.rand() < 1 - gating_percentage:
            gating_layer[i] = 1
        gating_task.append(gating_layer)
      gating_masks_per_task[num].append(gating_task)
      model_normal.layers[3].mask = gating_task[0]
      model_normal.layers[6].mask = gating_task[1]

    if use_EWC is True:
      if num == 0:
        list_of_accuracies, mean = train_normal.train(model_normal, epochs, train, test_tasks=test_tasks, XdG=use_XdG, gating=gating_masks_per_task)
      else:
        list_of_accuracies, mean = train_normal.train(model_normal, epochs, train, prior_weights=prior_weights_var, fisher_matrix=f_matrix, test_tasks=test_tasks, XdG=use_XdG, gating=gating_masks_per_task)
      prior_weights_var = model_normal.get_weights()
      ewc = EWC(model_normal, x_train_unbatched, num_ewc_sample=num_ewc_sample)
      f_matrix = ewc.get_fisher()

    elif use_SI is True:
      list_of_accuracies, mean = SI.train_SI(model_normal, train, test_tasks, XdG=use_XdG, gating=gating_masks_per_task)
      SI.reset_small_omega()

    else:
      list_of_accuracies, mean = train_normal.train(model_normal, epochs, train, test_tasks=test_tasks, XdG=use_XdG, gating=gating_masks_per_task)
    complete_mean.append(mean)
    list_of_accs.append(list_of_accuracies)

  return complete_mean
