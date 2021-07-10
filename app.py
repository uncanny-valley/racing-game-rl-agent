import base64
import json
import logging
import numpy as np
import os

from flask import Flask, request, jsonify
from multiexit import install, register

from action import Action
from agent import Agent
from reward import sum_rewards
from telemetry import Telemetry

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

np.random.seed(0)
app = Flask(__name__)
agent = Agent(checkpoints_dir=os.path.join(app.root_path, 'checkpoints'))

@app.route('/frame', methods=['POST'])
def next_action():
    metadata = json.loads(request.args.get('metadata'))

    # Positional telemetry for all 4 action-repeat sub-frames
    telemetries = [Telemetry.from_json(t) for t in metadata.get('buffer')]

    
    # Whether or not the frame is the initial state to an episode
    is_initial_frame = metadata.get('is_initial_frame')

    # The last action performed, derived from a mapping of input keys
    if is_initial_frame:
        action = None
    else:
        action = Action.from_array(keys=metadata.get('action'))

    # Whether or not the state is terminal (i.e. the episode has ended)
    # The vehicle drifting off-lane denotes termination.
    is_terminal = metadata.get('terminal')

    # Total reward summed over all action repeat sub-frames
    reward = sum_rewards(telemetries)

    # Pre-process frame data for ingestion
    # The resulting frame corresponds to the final, pre-processed sub-frame in our set of action repeat sub-frames
    data = request.get_json()
    state = np.fromstring(base64.decodestring(data.encode('utf-8')), dtype=np.uint8)
    state = state.reshape(84, 84, 3)

    a = agent.act(state, action, reward, is_terminal, is_initial_frame, telemetries)

    action = Action.random_action()
    return jsonify({
        'action': action.name,
        'keys': action.to_keys()})



@app.route('/reward', methods=['POST'])
def reward():
    """
    For human players that do not need instruction on the next action to take, but want to
    be evaluated by our reward function
    """
    metadata = json.loads(request.args.get('metadata'))

    # Positional telemetry for all 4 action-repeat sub-frames
    telemetries = [Telemetry.from_json(t) for t in metadata.get('buffer')]
    reward = sum_rewards(telemetries)

    return jsonify({'reward': reward})


install()

def exit_handler():
    response = str(input('Save weights? (Y / N):\n')).lower().strip()
    if response[0] == 'y':
        agent.save_model_weights()

register(exit_handler, shared=False)

app.run(host="0.0.0.0", port=8080, threaded=True)
