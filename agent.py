import numpy as np

import hyperparameters as hp
from action import Action
from experience_replay import ExperienceReplay

import tensorflow as tf

def create_q_model():
    inputs = tf.keras.layers.Input(shape=(84, 84, 3))

    # Convolutions on the frames on the screen
    layer1 = tf.keras.layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    layer2 = tf.keras.layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = tf.keras.layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

    layer4 = tf.keras.layers.Flatten()(layer3)

    layer5 = tf.keras.layers.Dense(512, activation="relu")(layer4)
    action = tf.keras.layers.Dense(Action.num_actions(), activation="linear")(layer5)

    return tf.keras.Model(inputs=inputs, outputs=action)


class Agent:
    def __init__(self,
            exp_replay_buffer_size=hp.EXPERIENCE_REPLAY_BUFFER_SIZE, 
            epsilon=hp.EPSILON):
        self._num_timesteps = -1
        self._epsilon = epsilon
        self._experience_replay = ExperienceReplay(size=exp_replay_buffer_size)
        self._model = create_q_model()
        self._model_target = create_q_model()
        self._loss_func = tf.keras.losses.Huber()
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

    def _is_ready_to_update_model(self):
        return self._num_timesteps % hp.UPDATE_MODEL_FREQUENCY == 0
    
    def _is_ready_to_update_target_model(self):
        return self._num_timesteps % hp.UPDATE_TARGET_MODEL_FREQUENCY == 0

    def act(self, state, action, reward, is_terminal, is_initial_frame, telemetries):
        self._num_timesteps += 1

        # self._experience_replay.add_transition(state, action, reward, next_state, is_terminal, is_initial_frame)
        # minibatch = self._experience_replay.sample_minibatch(batch_size=5)
        # print('minibatch', minibatch)
        
        # When the agent has no prior experience, choose a random action
        if self._num_timesteps <= 0:
            return Action.random_action()

        if np.random.rand() < self._epsilon:
            return Action.random_action()
        else:
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = self._model(state_tensor, training=False)
            print('action probs:', action_probs)
            # Take best action
            action = tf.argmax(action_probs[0]).numpy()
            print('action', action)

        if self._is_ready_to_update_model() and self._experience_replay.size > hp.EXPERIENCE_REPLAY_MINIBATCH_SIZE:
            batch = self._experience_replay.sample_minibatch(batch_size=hp.EXPERIENCE_REPLAY_MINIBATCH_SIZE)
            state_samples, action_samples, reward_samples, terminal_samples, next_state_samples = batch

        

        return Action.random_action()
