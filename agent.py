import datetime
import numpy as np
import os

import hyperparameters as hp
from action import Action
from experience_replay import ExperienceReplay

import tensorflow as tf


class Agent:
    def __init__(self,
            exp_replay_buffer_size=hp.EXPERIENCE_REPLAY_BUFFER_SIZE,
            minibatch_size=hp.EXPERIENCE_REPLAY_MINIBATCH_SIZE,
            discount_factor=0.99,
            checkpoint_date_to_load=None,
            checkpoints_dir=None):

        self._checkpoint_date_to_load = checkpoint_date_to_load
        self._checkpoints_dir = checkpoints_dir
        self._road_progress_history = np.array([], dtype=np.uint64)
        self._proportion_inlane_history = np.array([], dtype=np.float64)
        self._episodic_reward_history = np.array([], dtype=np.float64)
        self._num_timesteps = -1

        self._episodic_timestep = 0
        self._episode_number = 0

        self._discount_factor = discount_factor

        self._epsilon_min = 0.05
        self._epsilon_max = 1.0
        self._epsilon = 0.3
        self._epsilon_interval = self._epsilon_max - self._epsilon_min 
        self._epsilon_random_frames = 50000
        self._epsilon_greedy_frames = 1000000
        self._epsilon_decay_rate = self._epsilon_interval / self._epsilon_greedy_frames
        
        self._experience_replay = ExperienceReplay(size=exp_replay_buffer_size)
        self._minibatch_size = minibatch_size
        
        self._model = self._initialize_q_network()
        self._model_target = self._initialize_q_network()
        
        if self._checkpoint_date_to_load:
            self._load_model_weights()

        self._loss_func = tf.keras.losses.Huber()
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0)

        self._last_state = None
        self._init_time = datetime.datetime.now().strftime('%Y:%m:%d:%H:%M')

    def _initialize_q_network(self):
        inputs = tf.keras.layers.Input(shape=(84, 84, 3,))
        l1 = tf.keras.layers.Conv2D(32, 8, strides=4, activation='relu')(inputs)
        l2 = tf.keras.layers.Conv2D(64, 4, strides=2, activation='relu')(l1)
        l3 = tf.keras.layers.Conv2D(64, 3, strides=1, activation='relu')(l2)

        l4 = tf.keras.layers.Flatten()(l3)

        l5 = tf.keras.layers.Dense(512, activation='relu')(l4)
        action = tf.keras.layers.Dense(Action.num_actions(), activation='linear')(l5)
        return tf.keras.Model(inputs=inputs, outputs=action)

    def save_model_weights(self):
        dt = self._init_time
        model_checkpoint_name = os.path.join(self._checkpoints_dir, f'q-network-{dt}')
        model_target_checkpoint_name = os.path.join(self._checkpoints_dir, f'q-target-network-{dt}')

        self._model.save_weights(model_checkpoint_name)
        self._model_target.save_weights(model_target_checkpoint_name)

    def _load_model_weights(self):
        self._model.load_weights(f'q-network={self._checkpoint_date_to_load}')
        self._model_target.load_weights(f'q-target-network={self._checkpoint_date_to_load}')

    def _is_ready_to_update_model(self):
        return self._num_timesteps % hp.UPDATE_MODEL_FREQUENCY == 0
    
    def _is_ready_to_update_target_model(self):
        return self._num_timesteps % hp.UPDATE_TARGET_MODEL_FREQUENCY == 0

    def _is_agent_inlane(self, telemetry):
        return abs(telemetry.position_x) < 0.8

    def _sum_inlane(self, telemetries):
        return np.sum([self._is_agent_inlane(t) for t in telemetries])
        
    def _mean_average_inlane(self, telemetries):
        return self._sum_inlane(telemetries) / len(telemetries)

    def act(self, state, action, reward, is_terminal, is_initial_frame, telemetries):
        self._num_timesteps += 1

        if is_initial_frame:
            self._episodic_timestep = 0
            self._episodic_reward_history = np.append(self._episodic_reward_history, reward)
            self._proportion_inlane_history = np.append(self._proportion_inlane_history, self._mean_average_inlane(telemetries))
            self._road_progress_history = np.append(self._road_progress_history, 1)
        else:
            self._episodic_timestep += 1
            self._episodic_reward_history[-1] += reward
            self._road_progress_history[-1] += 1
            
            # Compute moving average of proportion of frames in-lane
            n, m = self._episodic_timestep + 1, len(telemetries)
            self._proportion_inlane_history[-1] = self._proportion_inlane_history[-1] * ((n - m) / n) + (self._sum_inlane(telemetries) / n)
        
        # When the agent has no prior experience, choose a random action
        if self._num_timesteps <= 0:
            return Action.random_action()

        print(f'[episode={self._episode_number}, episodic_ts={self._episodic_timestep}, total_ts={self._num_timesteps}, eps={self._epsilon}]: reward={reward}')
        if is_terminal:
            print(f'[episode={self._episode_number}]: road progress={self._road_progress_history[-1]}')
            if self._episode_number % 100 == 0:
                with open(f'data/{self._init_time}-episodic-reward-history.txt', 'ab') as f:
                    np.savetxt(f, self._episodic_reward_history)
                
                with open(f'data/{self._init_time}-road-progress-history.txt', 'ab') as f:
                    np.savetxt(f, self._road_progress_history)

                with open(f'data/{self._init_time}-proportion-inlane-history.txt', 'ab') as f:
                    np.savetxt(f, self._proportion_inlane_history)
            
                self._episodic_reward_history = []
                self._road_progress_history = []
                self._proportion_inlane_history = []

            self._episode_number += 1


        # If not the initial frame in an episode, add a transition (s, a, r, s', t) to replay memory
        if not is_initial_frame and self._last_state is not None:
            self._experience_replay.add_transition(self._last_state.copy(), action, reward, state, is_terminal, is_initial_frame)

        if self._is_ready_to_update_model() and len(self._experience_replay) > self._minibatch_size:
            batch = self._experience_replay.sample_minibatch(batch_size=self._minibatch_size)
            state_samples, action_samples, reward_samples, terminal_samples, next_state_samples, is_initial_state_samples = batch

            future_rewards = self._model_target.predict(next_state_samples)

            # Q = reward + discount factor * expected future reward
            updated_q_values = reward_samples + self._discount_factor * tf.reduce_max(future_rewards, axis=1)

            updated_q_values = updated_q_values * (1 - terminal_samples) - terminal_samples

            # Create a mask so we only calculate loss on the updated Q-values
            masks = tf.one_hot(action_samples, Action.num_actions())

            with tf.GradientTape() as tape:
                # Train the model on the states and updated Q-values
                q_values = self._model(state_samples)

                # Apply the masks to the Q-values to get the Q-value for action taken
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)

                # Calculate loss between new Q-value and old Q-value
                loss = self._loss_func(updated_q_values, q_action)
        
            # Backpropagation for loss
            grads = tape.gradient(loss, self._model.trainable_variables)
            self._optimizer.apply_gradients(zip(grads, self._model.trainable_variables))


        if self._is_ready_to_update_target_model():
            self._model_target.set_weights(self._model.get_weights())


        if self._num_timesteps < self._epsilon_random_frames or np.random.rand() < self._epsilon:
            next_action = Action.random_action()
        else:
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = self._model(state_tensor, training=False)
            next_action = Action(tf.argmax(action_probs[0]).numpy())

        self._epsilon -= self._epsilon_decay_rate
        self._epsilon = max(self._epsilon, self._epsilon_min)

        self._last_state = state.copy()

        return next_action
