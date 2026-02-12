""" 
File for creating the state representation and reward definition
"""

import sys
import optparse
import random
import traci
import time
import numpy as np
from xml.dom import minidom
from utils import *

# Force the state representation to use congestion length (halting vehicles)
config = import_train_configuration(config_file='training_settings.ini')
config['state_representation'] = 'congestion_length'

TL_list = {"3030": 'cluster_25579770_2633530003_2633530004_2633530005'}

def init_states(TL_list: dict) -> dict:
    junction_inits = dict.fromkeys(TL_list)
    for junction in TL_list:
        junction_inits[junction] = State_Observer(config['sumocfg_file_name'], junction, config['state_representation'], 'sum',
                                                  config['reward_definition'], 'sum')
    return junction_inits

def get_states(junction_list: dict) -> None:
    for junction in junction_list:
        junction_list[junction].append_states()

def get_current_reward(junction_list: dict) -> None:
    for junction in junction_list:
        junction_list[junction].get_rewards()

def get_state_size(junction_list: dict, num_programs: dict) -> dict:
    state_size_dict = dict.fromkeys(junction_list)
    for junction in junction_list:
        state_size = junction_list[junction].get_state_dimension()
        if state_size is None:
            raise ValueError(f"State dimension is None for junction: {junction}")
        state_size_dict[junction] = state_size + num_programs[junction]
    return state_size_dict

def return_states(action_dict: dict, junction_list: dict, program_dict: dict, num_programs: dict) -> dict:
    action_states_dict = dict.fromkeys(junction_list)
    for junction in junction_list:
        if action_dict[junction] == 1:
            aggregated_states = junction_list[junction].aggregate_states()
            one_hot = np.zeros(num_programs[junction])
            if len(one_hot) > 0:
                one_hot[program_dict[junction]] = 1
                aggregated_states_one_hot = [*aggregated_states, *one_hot]
                action_states_dict[junction] = aggregated_states_one_hot
            else:
                aggregated_states.append(traci.trafficlight.getPhase(junction))
                action_states_dict[junction] = aggregated_states
            junction_list[junction].clear_states()
        if action_dict[junction] == 0:
            action_states_dict[junction] = None
    return action_states_dict

def return_reward(action_dict: dict, junction_list: dict) -> dict:
    action_reward_dict = dict.fromkeys(junction_list)
    for junction in junction_list:
        if action_dict[junction] == 1:
            aggregated_rewards = junction_list[junction].aggregate_reward()
            action_reward_dict[junction] = aggregated_rewards
            junction_list[junction].clear_rewards()
        if action_dict[junction] == 0:
            action_reward_dict[junction] = None
    return action_reward_dict

def onehot_program(program_list: dict, num_programs: int):
    programs = np.array(program_list)
    programs_ = np.zeros((programs.size, num_programs))
    programs_[np.arange(programs.size), programs] = 1
    return programs_.tolist()

class State_Observer():
    def __init__(self, sumo_net, junction_name, state_representation, aggregation_method,
                 reward_definition, reward_aggregation):
        self.junction_name = junction_name
        self.state_representation = state_representation
        self.sumo_net = sumo_net
        self.aggregation_method = aggregation_method
        self.states = list()
        self.reward = 0
        self.reward_counter = 0
        self.reward_definition = reward_definition
        self.reward_aggregation = reward_aggregation

        self.incLanes_list = list(traci.trafficlight.getControlledLanes(junction_name))
        self.incLanes_list = [i for i in self.incLanes_list if i[0] != ":"]
        self.incLanes_list = list(dict.fromkeys(self.incLanes_list))

    def get_state(self) -> list:
        # Modified: state = number of stopping vehicles per lane (congestion length)
        state = []
        for lane in self.incLanes_list:
            state.append(traci.lane.getLastStepHaltingNumber(lane))
        return state

    def get_state_dimension(self) -> int:
        return len(self.incLanes_list)

    def aggregate_states(self) -> list:
        if not self.states:
            return [0] * len(self.incLanes_list)
        states_arr = np.array(self.states)
        if self.aggregation_method == "mean":
            return list(np.mean(states_arr, axis=0))
        elif self.aggregation_method == "sum":
            return list(np.sum(states_arr, axis=0))
        else:
            raise ValueError("Invalid aggregation method")

    def append_states(self):
        self.states.append(self.get_state())

    def clear_states(self):
        self.states = []

    def get_rewards(self):
        self.reward_counter += 1
        waiting_total = 0
        if self.reward_definition == "waiting":
            for lane in self.incLanes_list:
                waiting_total += traci.lane.getLastStepHaltingNumber(lane)
        elif self.reward_definition == "waiting_fast":
            for veh_id in traci.simulation.getDepartedIDList():
                traci.vehicle.subscribe(veh_id, [traci.constants.VAR_LANE_ID, traci.constants.VAR_SPEED])
            result = traci.vehicle.getAllSubscriptionResults()
            for lane in self.incLanes_list:
                waiting_total += sum(result[value][81] == lane and result[value][64] < 0.1 for value in result)
        else:
            raise ValueError("Invalid reward definition")

        self.reward += -waiting_total

    def aggregate_reward(self):
        if self.reward_counter == 0:
            return 0
        if self.reward_aggregation == "mean":
            return self.reward / self.reward_counter
        elif self.reward_aggregation == "sum":
            return self.reward
        else:
            raise ValueError("Invalid reward aggregation type")

    def clear_rewards(self):
        self.reward = 0
        self.reward_counter = 0

    def get_incLanes(self):
        print(self.incLanes_list)

    def get_more_lanes(self, init_lanes_list):
        add_lanes_list = list()
        for lane_index, init_lane in enumerate(init_lanes_list):
            init_edge = traci.lane.getEdgeID(init_lane)
            add_lanes_list.append(init_lane)
            xml_net = minidom.parse(self.sumo_net)
            connection_list = xml_net.getElementsByTagName('connection')
            for s in connection_list:
                if s.attributes['to'].value == init_edge and s.attributes['toLane'] == init_lane[-1]:
                    more_edge = s.attributes['from'].value
                    more_lane_id = s.attributes['fromLane'].value
                    add_lanes_list[lane_index].extend(more_edge + "_" + more_lane_id)



