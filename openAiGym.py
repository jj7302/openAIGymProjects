import gym
from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(4,)),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(2)
])


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

class Sample():
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.observations = [] #holds all of the observations of successful random approached
        self.actions = [] #holds all of the resulting actions
        self.steps = [] #number of steps corresponding to successful samples
        self.all_steps = []
        self.selected_average_steps = 0
        self.average_steps = 0

    def compute_average_steps(self):
        self.average_steps = 0
        for i in self.all_steps:
            self.average_steps += i
        self.average_steps = self.average_steps / len(self.all_steps)

        self.selected_average_steps = 0
        for i in self.steps:
            self.selected_average_steps += i
        if len(self.steps) > 0:
            self.selected_average_steps = self.selected_average_steps / len(self.steps)

    def generate_random_sample(self, minimum_steps=50, samples=50000, visualize=False):
        for _ in range(samples):
            observation = self.env.reset()
            action_list = []
            observation_list = []
            for t in range(1000):
                if visualize:
                    self.env.render()
                action = self.env.action_space.sample()
                action_list.append(action)
                observation_list.append(observation)
                observation, reward, done, info = self.env.step(action)
                if done:
                    if t > minimum_steps:
                        for item in observation_list:
                            self.observations.append(item)
                        for item in action_list:
                            self.actions.append(item)
                        self.steps.append(t)
                    self.all_steps.append(t)
                    break

        self.env.close()
        self.compute_average_steps()

    def generate_sample_from_model(self, minimum_steps=50, samples=4000, visualize=False, model=None):
        for _ in range(samples):
            if _ % 100 == 0:
                print(_)
            observation = self.env.reset()
            action_list = []
            observation_list = []
            for t in range(1000):
                if visualize:
                    self.env.render()
                action = model.predict([observation])[0]
                action_list.append(action)
                observation_list.append(observation)
                observation, reward, done, info = self.env.step(action)
                if done:
                    if t > minimum_steps:
                        for item in observation_list:
                            self.observations.append(item)
                        for item in action_list:
                            self.actions.append(item)
                        self.steps.append(t)
                    self.all_steps.append(t)
                    break

        self.env.close()
        self.compute_average_steps()

from sklearn.neural_network import MLPClassifier
x = Sample()
x.generate_random_sample()
print(x.average_steps)
print(x.selected_average_steps)
xTrain, xTest, yTrain, yTest = train_test_split(x.observations, x.actions)

num_training_cycles = 30
min_steps = 80
for i in range(num_training_cycles):
    model = MLPClassifier(solver='lbfgs', random_state=2, hidden_layer_sizes=[80, 160, 80])
    model.fit(xTrain, yTrain)
    #print(model.predict(xTest))
    print(model.score(xTest, yTest))
    samp = Sample()
    samp.generate_sample_from_model(model=model, minimum_steps=min_steps)
    print(samp.average_steps)
    print(samp.selected_average_steps)
    xTrain, xTest, yTrain, yTest = train_test_split(samp.observations, samp.actions)
    min_steps += 40
#model.fit(xTrain, yTrain, epochs=10)
#print(model.predict())