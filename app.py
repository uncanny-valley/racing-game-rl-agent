import base64
import json
import numpy as np

from flask import Flask, request, jsonify

from action import Action
from agent import Agent
from reward import sum_rewards
from telemetry import Telemetry


app = Flask(__name__)

@app.route('/frame', methods=['POST'])
def next_action():
    metadata = json.loads(request.args.get('metadata'))

    # Positional telemetry for all 4 action-repeat sub-frames
    telemetries = [Telemetry.from_json(t) for t in metadata.get('buffer')]

    # The last action performed, derived from a mapping of input keys
    action = Action.from_array(keys=metadata.get('action'))

    # Whether or not the state is terminal (i.e. the episode has ended)
    # The vehicle drifting off-lane denotes termination.
    is_terminal = metadata.get('terminal')

    # Whether or not the frame is the initial state to an episode
    is_initial_frame = metadata.get('is_initial')

    # Total reward summed over all action repeat sub-frames
    reward = sum_rewards(telemetries)

    # Pre-process frame data for ingestion
    # The resulting frame corresponds to the final, pre-processed sub-frame in our set of action repeat sub-frames
    data = request.get_json()
    state = np.fromstring(base64.decodestring(data.encode('utf-8')), dtype=np.uint8)
    state = state.reshape(84, 84, 3)
    
    agent = Agent()
    a = agent.act(state, action, reward, is_terminal, is_initial_frame, telemetries)
    print('action:', a)

    action = Action.random_action()
    return jsonify({
        'action': action.name,
        'keys': action.to_keys()})
