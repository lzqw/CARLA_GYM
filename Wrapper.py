from __future__ import print_function

import copy
import csv
import logging
import math
import socket
import subprocess
import threading
import time
from distutils.version import LooseVersion
import carla
import cv2
import gym
import sys
import os

import matplotlib.pyplot as plt
import py_trees
import torch

import numpy as np
import pkg_resources
from PIL import Image, ImageDraw
from typing import Any, Dict
from PythonAPI.carla.agents.navigation.local_planner import RoadOption

from utils.logger import logger,setup_logger
from utils.planner import RoutePlanner

sys.path.append(os.path.join(os.path.dirname(__file__), 'leaderboard'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'scenario_runner'))

# 设置carla python api的路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'PythonAPI', 'carla'))
os.environ['SCENARIO_RUNNER_ROOT']=os.path.join(os.path.dirname(__file__), 'scenario_runner')

SERVER_BINARY = os.environ.get("CARLA_SERVER", os.path.expanduser(
    "/home/lzqw/PycharmProject/Carla-RL/CARLA_Leaderboard_20/CarlaUE4.sh"))

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog
from leaderboard.autoagents.agent_wrapper import AgentError
from leaderboard.scenarios.route_scenario import RouteScenario
from srunner.scenariomanager.traffic_events import TrafficEventType
from leaderboard.utils.priority_route_indexer import PriorityRouteIndexer
from leaderboard.utils.route_indexer import RouteIndexer
from leaderboard.utils.route_manipulation import downsample_route
from leaderboard.envs.sensor_interface import SensorInterface, CallBack, OpenDriveMapReader, SpeedometerReader, \
    SensorConfigurationInvalid
from srunner.scenariomanager.timer import GameTime

"""
from test
"""
from config.utils.utils import Config,check_exist


logging.basicConfig(level=logging.DEBUG)
sensors_to_icons = {
    'sensor.camera.rgb': 'carla_camera',
    'sensor.lidar.ray_cast': 'carla_lidar',
    'sensor.other.radar': 'carla_radar',
    'sensor.other.gnss': 'carla_gnss',
    'sensor.other.imu': 'carla_imu',
    'sensor.opendrive_map': 'carla_opendrive_map',
    'sensor.speedometer': 'carla_speedometer',
    'sensor.other.obstacle': 'carla_obstacle'
}



class EnvWrapper(gym.Env):
    def __init__(self, config,**kwargs)->None:
        self.rank = config.rank
        self.debug_mode = config.debug
        self.cuda_id =config.cuda_id#new
        self.render=config.render#new
        self.fixed_port=config.fixed_port#new
        self.quality_level=config.quality_level#new
        self.frame_rate = config.frame_rate
        self._timeout = config.timeout
        self.vehicle_block_time = config.vehicle_block_time
        self._step = 0
        self._min_speed = config.min_speed
        self._max_speed = config.max_speed
        self._target_speed = config.target_speed
        self._max_degree = config.max_degree
        self._seq_length = config.seq_length
        self.pre_theta = 0
        self.sensor_interface = None
        self.agent_sensor_list = config.sensor_list
        self.training = config.training
        self._sensors_list = []
        self.scenario_class = None
        self.scenario = None
        self.scenario_tree = None
        self._history_tick_data = dict(
            rgb=[],
            measurements=[],
            route_fig=[],
        )

        self.world=None

        self.sensors = None
        self._vehicle_lights = carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam
        # self._init_server()
        self.port=2000
        self.client = carla.Client(config.host, self.port)
        self.client.set_timeout(config.client_timeout)

        trafficManagerPort = self._get_tcp_port()

        self.traffic_manager = self.client.get_trafficmanager(trafficManagerPort)
        self.trafficManagerPort = trafficManagerPort
        self.trafficManagerSeed = config.rank


        # self.world = self.client.load_world('Town13')
        # time.sleep(100)
        # print("111111111111111111111111111111111111111111111111")
        # settings = self.world.get_settings()
        # settings.fixed_delta_seconds = 1.0 / self.frame_rate
        # settings.synchronous_mode = True
        # self.world.apply_settings(settings)
        #
        # self.world.reset_all_traffic_lights()
        # CarlaDataProvider._training = self.training
        # CarlaDataProvider.set_client(self.client)
        # CarlaDataProvider.set_world(self.world)
        # CarlaDataProvider.set_traffic_manager_port(self.trafficManagerPort)
        # self.traffic_manager.set_synchronous_mode(True)
        # self.traffic_manager.set_random_device_seed(self.trafficManagerSeed)
        # if CarlaDataProvider.is_sync_mode():
        #     self.world.tick()
        # else:
        #     self.world.wait_for_tick()


        dist = pkg_resources.get_distribution("carla")
        if dist.version != 'leaderboard':
            if LooseVersion(dist.version) < LooseVersion('0.9.14'):
                raise ImportError("CARLA version 0.9.14 or newer required. CARLA version found: {}".format(dist))

        if config.training:
            root_path = config.root_path
            local_time = str(time.strftime("%m-%d/%H-%M-%S", time.localtime()))
            root_path = os.path.join(root_path, local_time)
            check_exist(root_path)
            leaderboard_result_file_path = os.path.join(root_path, str(config.rank))
            if not os.path.exists(leaderboard_result_file_path):
                os.mkdir(leaderboard_result_file_path)
            if self.rank == 0:
                logger.log('training results were saved in {} '.format(leaderboard_result_file_path))
            leaderboard_result_file_path = os.path.join(leaderboard_result_file_path, 'file')
            if not os.path.exists(leaderboard_result_file_path):
                os.mkdir(leaderboard_result_file_path)
            average_completion_ratio_path = os.path.join(leaderboard_result_file_path, 'completion_ratio.csv')
            work_dir = os.path.join(root_path, str(config.rank))
            self.work_dir = work_dir
        else:
            if not os.path.exists(config.pretrained_path):
                print('Error: pretrained model path {} does not exist!'.format(config.pretrained_path))
                exit(-1)
            leaderboard_result_file_path = os.path.join(config.pretrained_path, 'eval')
            check_exist(leaderboard_result_file_path)
            average_completion_ratio_path = os.path.join(leaderboard_result_file_path, 'eval_completion_ratio.csv')
            root_path = leaderboard_result_file_path
            self.work_dir = root_path

        self.average_completion_ratio_path = average_completion_ratio_path

        with open(self.average_completion_ratio_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["route_index", "completion_ratio"])

        if config.training:
            check_exist(self.work_dir)
            log_file_name = os.path.join(self.work_dir, 'file')
            check_exist(log_file_name)

        setup_logger(self.work_dir, log_dir=self.work_dir)

        self.route_name = -1
        self.completion_ratio = 0
        self.error_message = ""
        self.loop = 0
        self.route_indexer = None
        # if config.training is False:
        #     self.route_indexer = RouteIndexer(config.routes, config.scenarios, config.amount)
        # else:
        #     if config.route_indexer == "priority":
        #         self.route_indexer = PriorityRouteIndexer(config.routes, config.scenarios, config.amount)
        #     else:
        #         print('Error no such route_indexer as {}.'.format(config.route_indexer))
        #         exit(-1)
        self.route_indexer = RouteIndexer('/home/lzqw/PycharmProject/Carla-RL/CARLA_GYM/scenario_runner/srunner'
                                          '/data/test.xml', 100, '')

        self.ego_vehicles = None

        self._timestamp_last_run = 0.0
        # GameTime.restart()
        self.event_num = np.zeros(7)
        self.begin = True
        self.action_space = gym.spaces.Box(low=np.array([-1, 0, 0]), high=np.array([1, 1, 1]), dtype=np.float)
        self.error_message = ""

    def is_avalable(self):
        return self.route_indexer[0].peek()
    def compute_reward(self, speed, dis, theta, new_event_list, obstacle, max_block_time=400):
        event_reward = 0
        throttle_event_reward = 0
        steer_event_reward = 0
        target_reached = False
        done = 0
        throttle_done = 0
        steer_done = 0
        error_message = ""
        if self.begin is False:
            for event in new_event_list:
                if event.get_type() == TrafficEventType.COLLISION_STATIC:
                    error_message = "collision static"
                    steer_event_reward -= 1
                    # event_reward -= 2
                    steer_done = 1
                    if self.training:
                        done = 1
                elif event.get_type() == TrafficEventType.COLLISION_PEDESTRIAN or event.get_type() == TrafficEventType.COLLISION_VEHICLE:
                    # if event.get_dict():
                    #     intensity = event.get_dict()['intensity']
                    #
                    throttle_event_reward -= 1
                    throttle_done = 1
                    done = 1
                    # if intensity >= 400:
                    if event.get_type() == TrafficEventType.COLLISION_PEDESTRIAN:
                        error_message = "collision pedestrians!"
                    else:
                        error_message = "collision vehicles!"
                elif event.get_type() == TrafficEventType.VEHICLE_BLOCKED:
                    error_message = "vehicle blocked"
                    done = 1
                    throttle_done = 1
                    # event_reward -= 1
                    throttle_event_reward -= 1
                elif event.get_type() == TrafficEventType.ROUTE_DEVIATION:
                    error_message = "route deviation"
                    done = 1
                    # event_reward -= 1
                    steer_event_reward -= 1
                    steer_done = 1
                elif event.get_type() == TrafficEventType.ROUTE_COMPLETED:
                    steer_done = 1
                    throttle_done = 1
                    error_message = "success"
                    steer_event_reward += 5
                    throttle_event_reward += 5
                    target_reached = True
                    done = 1
                elif event.get_type() == TrafficEventType.ROUTE_COMPLETION:
                    if not target_reached:
                        if event.get_dict():
                            score_route = event.get_dict()['route_completed']
                        else:
                            score_route = 0
                        error_message = "route completion with {}".format(score_route)

                        event_reward += 5 * score_route
                    done = 1
                elif event.get_type() == TrafficEventType.OUTSIDE_ROUTE_LANES_INFRACTION:
                    error_message = "outside route!"
                    done = 1
                    steer_event_reward -= 1
                    steer_done = 1
        else:
            self.begin = False

        # ================= theta_reward: 0-1 =====================
        degree = abs(180 * theta / np.pi)
        if self.in_turn:
            degree = max(0, degree - 30)

        theta_reward = max(0, 1 - degree / self._max_degree)
        if speed > self._max_speed:
            # event_reward -= 2
            throttle_event_reward -= 1
            throttle_done = 1
            if self.training:
                done = True
                error_message = 'exceed speed'

        detect_obstacle = obstacle > -1 and obstacle < 12
        if detect_obstacle:  # detect obstacles
            self.last_event_timestamp = self._step
            target_speed = max(0, (obstacle - 5))
            speed_reward = 1 - max((speed - target_speed), 0) / (self._max_speed - target_speed)
            if obstacle < 5:
                if speed > 0.1:
                    speed_reward = -1
                else:
                    speed_reward = 1

        else:
            if speed < self._min_speed:
                speed_reward = speed / self._min_speed
            elif speed > self._target_speed:
                speed_reward = max(0, 1 - (speed - self._target_speed) / (self._max_speed - self._target_speed))
            else:
                speed_reward = 1

        # # ================ deviation reward: 0 ~ -1 ============================
        if self.in_turn or self.near_command != RoadOption.LANEFOLLOW:
            D_max = 5
        else:
            D_max = 2.5
        if self.training is False:
            D_max = 10

        deviation_reward = max(0.0, 1.0 - dis / D_max)
        if speed < 1 and (self._step - self.last_event_timestamp) > max_block_time:
            self.last_event_timestamp = self._step
            done = 1
            throttle_event_reward -= 2
            throttle_done = 1
            error_message = "vehicle blocked"

        if len(new_event_list) > 0 or speed > 1:
            self.last_event_timestamp = self._step
        throttle_reward = speed_reward + throttle_event_reward
        steer_reward = (deviation_reward + theta_reward) / 2 + steer_event_reward
        return torch.tensor([steer_reward, throttle_reward]), done, error_message, [steer_done, throttle_done]
    def step(self, output_action):
        self._step += 1
        control = carla.VehicleControl()
        # output_action = output_action.squeeze()
        control.steer = output_action[0]
        control.throttle = output_action[1]
        control.brake = output_action[2]
        control.manual_gear_shift = False

        self.ego_vehicles[0].apply_control(control)
        # self.scenario.build_scenarios(self.ego_vehicles[0])
        # self.scenario.spawn_parked_vehicles(self.ego_vehicles[0])

        self.pre_control = [control.steer, control.throttle, control.brake]

        py_trees.blackboard.Blackboard().set("AV_control", control, overwrite=True)
        self.scenario_tree.tick_once()
        if self.scenario_tree.status != py_trees.common.Status.RUNNING:
            self._running = False
        self.spectator = CarlaDataProvider.get_world().get_spectator()
        ego_trans = self.ego_vehicles[0].get_transform()
        self.spectator.set_transform(carla.Transform(ego_trans.location + carla.Location(z=50),
                                                carla.Rotation(pitch=-90)))
        self.world.tick(self._timeout)

        # while not self._tick_scenario():
        #     continue

        tick_data = self._tick()
        gps = self._get_position(tick_data)
        near_node, near_command, route_list = self._waypoint_planner.run_step(gps)
        self.near_command = near_command
        tick_data['command'] = int(near_command.value) - 1

        tick_data, dis, theta, in_corner = self._draw_route(tick_data, route_list)
        tick_data['last_measurements'] = [tick_data['speed'] / self._max_speed, dis / 3.,
                                          abs(180 * theta / np.pi) / 90.]
        max_block_time = self.vehicle_block_time if self.training else 800

        rewards, done, error_message, action_done = self.compute_reward(tick_data['speed'], dis, theta,
                                                                        tick_data['new_event_list'],
                                                                        tick_data['obstacle'],
                                                                        max_block_time=max_block_time)
        if done:
            self._running=False
            self.error_message = error_message
        info = {'action_done': action_done}

        while len(self._history_tick_data['rgb']) >= self._seq_length:
            del self._history_tick_data['rgb'][0]

        self._history_tick_data['rgb'].append(copy.deepcopy(tick_data['last_rgb']))

        tick_data['rgb'] = np.array(self._history_tick_data['rgb'])

        while len(self._history_tick_data['measurements']) >= self._seq_length:
            del self._history_tick_data['measurements'][0]
        self._history_tick_data['measurements'].append(copy.deepcopy(tick_data['last_measurements']))
        tick_data['measurements'] = np.array(self._history_tick_data['measurements'])

        while len(self._history_tick_data['route_fig']) >= self._seq_length:
            del self._history_tick_data['route_fig'][0]
        self._history_tick_data['route_fig'].append(copy.deepcopy(tick_data['last_route_fig']))
        tick_data['route_fig'] = np.array(self._history_tick_data['route_fig'])

        if done:
            self.cleanup_scenario()
            self._cleanup()
        return tick_data, rewards, done, info

    def compute_duration_time(self):
        """
        Computes system and game duration times
        """
        self.end_system_time = time.time()
        self.end_game_time = GameTime.get_time()

        self.scenario_duration_system = self.end_system_time - self.start_system_time
        self.scenario_duration_game = self.end_game_time - self.start_game_time

    def cleanup_scenario(self):

        if self._watchdog:
            self._watchdog.stop()

        if self._agent_watchdog:
            self._agent_watchdog.stop()

        self.compute_duration_time()

        self.sensor_interface.destroy()
        self.sensor_interface = None

        if self.get_running_status():
            if self.scenario is not None:
                self.scenario.terminate()
                for criterion in self.scenario.get_criteria():
                    if criterion.name == "RouteCompletionTest":
                        self.completion_ratio = criterion.actual_value
                        with open(self.average_completion_ratio_path, 'a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([self.route_name, self.completion_ratio])
                        if self.rank == 0:
                            logger.log('route : {}, completion_ratio:{:.2f}, terminate due to {}.\n'.format(self.route_name,
                                                                                                       self.completion_ratio,
                                                                                                       self.error_message))
        self.scenario.remove_all_actors()
        self.scenario = None
        self.scenario_tree = None
        self.scenario_class = None
        self._running = False
        self._scenario_thread.join()
        self._scenario_thread = None

        for i, _ in enumerate(self._sensors_list):
            if self._sensors_list[i] is not None:
                self._sensors_list[i].stop()
                self._sensors_list[i].destroy()
                self._sensors_list[i] = None
        # Tick once to destroy the sensors
        CarlaDataProvider.get_world().tick()
        self._sensors_list = []

        # Make sure no sensors are left streaming
        alive_sensors = self.world.get_actors().filter('*sensor*')
        for sensor in alive_sensors:
            if sensor.is_listening:
                sensor.stop()
            sensor.destroy()


    def build_scenarios_loop(self, debug):
        """
        Keep periodically trying to start the scenarios that are close to the ego vehicle
        Additionally, do the same for the spawned vehicles
        """
        while self._running:
            self.scenario.build_scenarios(self.ego_vehicles[0], debug=debug)
            self.scenario.spawn_parked_vehicles(self.ego_vehicles[0])
            time.sleep(1)

    def _cleanup(self):
        CarlaDataProvider.cleanup()

        try:
            if self.ego_vehicles is not None:
                self.ego_vehicles = None
        except Exception as e:
            print("\n\033[91mFailed to stop the agent:")


        self._timestamp_last_run = 0.0
        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = 0.0
        self.start_game_time = 0.0
        self.end_system_time = 0.0
        self.end_game_time = 0.0

        self._spectator = None
        self._watchdog = None
        self._agent_watchdog = None


    def reset(self):
        self._step = 0

        # ==================== get next config ===========================
        while self.route_indexer.peek():
            config = self.route_indexer.get_next_config()
            self._load_and_wait_for_world(config.town)
            # else:
            #     self.client.reload_world(reset_settings=False)
            # time.sleep(3)
            # config.agent = self._agent
            self.route_name = int(config.name.strip().split('_')[-1])

            # =================== get new route scenario and ego vehicle =====
            scenario = RouteScenario(world=self.world, config=config,
                                     debug_mode=self.debug_mode)
            GameTime.restart()
            if scenario.ego_vehicles is not None:
                self.route_index = config.index
                self.set_global_plan(scenario.gps_route, scenario.route)
                self.ego_vehicles = scenario.ego_vehicles
                self.scenario_class = scenario
                self.scenario = scenario
                self.scenario_tree = scenario.scenario_tree
                self.other_actors = scenario.other_actors
                self._spectator = CarlaDataProvider.get_world().get_spectator()
                self.sensor_interface = SensorInterface()
                self.setup_sensors(self.agent_sensor_list, self.debug_mode > 1)
                break

        self.world.tick()
        # =================== setup sensors ============================
        self.start_system_time = time.time()
        self.start_game_time = GameTime.get_time()
        self._running=True
        # Detects if the simulation is down
        self._watchdog = Watchdog(self._timeout)
        self._watchdog.start()
        # Stop the agent from freezing the simulation
        self._agent_watchdog = Watchdog(self._timeout)
        self._agent_watchdog.start()

        # self.scenario.build_scenarios(self.ego_vehicles[0])
        # self.scenario.spawn_parked_vehicles(self.ego_vehicles[0])
        self._scenario_thread = threading.Thread(target=self.build_scenarios_loop, args=(-1,))
        self._scenario_thread.start()

        self.event_num = np.zeros(7)

        self._vehicle = CarlaDataProvider.get_hero_actor()
        try:
            self.world = self._vehicle.get_world()
        except Exception as e:
            print('Unable to get world of vehicle')

        self._map = self.world.get_map()
        self.in_turn = False
        self.turn_first_node = None
        self.turn_last_node = None
        self.first_direction = 0
        self.last_direction = 0
        self.begin = True
        self.last_event_timestamp = 0
        self.pre_theta = 0

        # while not self._tick_scenario():
        #     continue
        tick_data = self._tick()
        gps = self._get_position(tick_data)
        near_node, near_command, route_list = self._waypoint_planner.run_step(gps)
        self.near_command = near_command
        tick_data['command'] = int(near_command.value) - 1
        tick_data, dis, theta, in_corner = self._draw_route(tick_data, route_list)
        # todo: speed: vehicle speed, dis: distance from vehicle to route, theta: degree between vehicle and route
        tick_data['last_measurements'] = [tick_data['speed'] / self._max_speed, dis / 3.,
                                          abs(180 * theta / np.pi) / 90.]
    def _load_and_wait_for_world(self, town_name):
        """
        加载一个新的CARLA世界，不改变设置，并提供数据给CarlaDataProvider
        """
        logging.debug("loading")
        self.world = self.client.load_world(town_name, reset_settings=False)
        logging.debug("finish")
        # Large Map settings are always reset, for some reason
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 1.0 / self.frame_rate
        settings.synchronous_mode = True
        self.world.apply_settings(settings)

        self.world.reset_all_traffic_lights()

        CarlaDataProvider._training = self.training
        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_traffic_manager_port(self.trafficManagerPort)
        self.traffic_manager.set_synchronous_mode(True)
        CarlaDataProvider.set_world(self.world)

        self.traffic_manager.set_random_device_seed(self.trafficManagerSeed)
        # Wait for the world to be ready
        self.world.tick()
        map_name = CarlaDataProvider.get_map().name.split("/")[-1]

    def _draw_route(self, tick_data, route_list, pixels_per_meter=3.66, size_x=144, size_y=256,
                    color=(255, 0, 0)):
        route_fig = np.zeros((256, 144), dtype=np.uint8)
        route_fig = Image.fromarray(route_fig)
        route_draw = ImageDraw.Draw(route_fig)

        color = 255

        compass = tick_data['compass']
        compass = 0.0 if np.isnan(compass) else compass
        compass = compass + np.pi / 2
        R = np.array([
            [np.cos(compass), -np.sin(compass)],
            [np.sin(compass), np.cos(compass)],
        ])
        pos = self._get_position(tick_data)
        far_node = None

        turn_pre_node = route_list[0]
        direction = 0
        pep_dis = np.inf
        ori_pep_dis = np.inf
        gps_x = size_x / 2
        gps_y = size_y / 2

        zeros_node_x, zeros_node_y = pixels_per_meter * (R.T.dot(route_list[0] - pos))
        zeros_node_x += size_x / 2


        zeros_node_y += size_y / 2
        turn_first_waypoint = turn_last_waypoint = None

        # forward_dis = self.compute_distance(pos, route_list[0])
        for i in range(1, len(route_list)):
            cur_node = route_list[i]
            pre_node = route_list[i - 1]
            _pre_x, _pre_y = pixels_per_meter * (R.T.dot(pre_node - pos))
            pre_x = _pre_x + size_x / 2
            pre_y = _pre_y + size_y / 2

            _cur_x, _cur_y = pixels_per_meter * (R.T.dot(cur_node - pos))
            cur_x = _cur_x + size_x / 2
            cur_y = _cur_y + size_y / 2
            route_draw.line((_pre_x + size_x / 2, _pre_y + size_y / 2, _cur_x + size_x / 2, _cur_y + size_y / 2),
                            color, width=15)
            if i == 1:
                turn_first_waypoint = [pre_x, pre_y]
            if i == len(route_list) - 1:
                turn_last_waypoint = [cur_x, cur_y]
            if math.sqrt((zeros_node_y - cur_y) ** 2 + (zeros_node_x - cur_x) ** 2) > 1e-3 and pep_dis == np.inf:
                ori_pep_dis = abs(
                    ((cur_y - zeros_node_y) * (gps_x - zeros_node_x) - (cur_x - zeros_node_x) * (
                            gps_y - zeros_node_y)) / math.sqrt(
                        (cur_y - zeros_node_y) ** 2 + (cur_x - zeros_node_x) ** 2))
                pep_dis = abs(
                    ((cur_node[1] - route_list[0][1]) * (pos[0] - route_list[0][0]) - (
                            cur_node[0] - route_list[0][0]) * (
                             pos[1] - route_list[0][1])) / math.sqrt(
                        (cur_node[1] - route_list[0][1]) ** 2 + (cur_node[0] - route_list[0][0]) ** 2))

            if abs(route_list[i][0] - route_list[0][0]) + abs(
                    route_list[i][1] - route_list[0][1]) > 1e-3 and far_node is None:
                far_node = route_list[i]

            turn_cur_node = route_list[i]
            if self.in_turn is False:
                if abs(turn_cur_node[0] - turn_pre_node[0]) < 1 or abs(turn_cur_node[1] - turn_pre_node[1]) < 1:
                    continue
                else:
                    if abs(turn_cur_node[0] - turn_pre_node[0]) < abs(turn_cur_node[1] - turn_pre_node[1]):
                        direction = 0
                    else:
                        direction = 1
                    if self.turn_first_node is None:
                        self.first_direction = direction
                        self.turn_first_node = turn_cur_node
                    else:
                        self.last_direction = direction
                        self.turn_last_node = turn_cur_node
                turn_pre_node = turn_cur_node
        if pep_dis == np.inf or np.isnan(pep_dis):
            pep_dis = 0
        route_fig = np.array(route_fig)
        tick_data['last_route_fig'] = route_fig

        theta, distance = self.get_theta(far_node, route_list[0], pos, tick_data, len(route_list), pixels_per_meter)
        if len(route_list) == 2:
            distance = pep_dis

        if self.turn_first_node is not None and self.turn_last_node is not None:
            if self.first_direction == 0:
                turn_middle_node = [self.turn_last_node[0], self.turn_first_node[1]]
            else:
                turn_middle_node = [self.turn_first_node[0], self.turn_last_node[1]]
            turn_dis = self.compute_distance(turn_middle_node, pos)
            max_dis = max(self.compute_distance(turn_middle_node, self.turn_first_node),
                          self.compute_distance(turn_middle_node, self.turn_last_node))

            if turn_dis < max_dis + 6:
                self.in_turn = True
            elif self.in_turn == True:
                self.in_turn = False
                self.turn_first_node = None
                self.turn_last_node = None
                self.first_direction = 0
                self.last_direction = 0
        return tick_data, distance, theta, self.in_turn
    def _get_position(self, tick_data):
        gps = tick_data['gps']
        gps = (gps - self._waypoint_planner.mean) * self._waypoint_planner.scale
        return gps

    def get_running_status(self):
        """
        returns:
           bool: False if watchdog exception occured, True otherwise
        """
        if self._watchdog:
            return self._watchdog.get_status()
        return True

    def _tick(self):

        if self._running and self.get_running_status():
            CarlaDataProvider.get_world().tick(self._timeout)
        timestamp = CarlaDataProvider.get_world().get_snapshot().timestamp
        if self._timestamp_last_run < timestamp.elapsed_seconds and self._running:
            self._timestamp_last_run = timestamp.elapsed_seconds

            self._watchdog.update()
            # Update game time and actor information
            GameTime.on_carla_tick(timestamp)
            CarlaDataProvider.on_carla_tick()
            self._watchdog.pause()
        try:
            self._agent_watchdog.resume()
            self._agent_watchdog.update()
            CarlaDataProvider.update_ego(self.ego_vehicles[0])
            data = self.sensor_interface.get_data(GameTime.get_frame())
            self._agent_watchdog.pause()
        # Special exception inside the agent that isn't caused by the agent
        except Exception as e:
            raise AgentError(e)

        self._watchdog.resume()
        self.sensor_interface.clear_obstacle('obstacle')
        node_cnt = 0
        new_event_list = []
        for node in self.scenario.get_criteria():
            if node.events:
                event_cnt = 0
                for event in node.events:
                    event_cnt += 1
                    if event_cnt > self.event_num[node_cnt]:
                        new_event_list.append(event)
                        self.event_num[node_cnt] += 1
            node_cnt += 1

        lidar= data['LIDAR']
        rgb = cv2.cvtColor(data['rgb'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        plt.clf()
        plt.imshow(rgb)
        plt.pause(0.1)
        gps = data['gps'][1][:2]
        full_gps = data['gps'][1][:3]
        target_diff = 0
        speed = data['speed'][1]['speed']
        if np.isnan(speed):
            speed = 0
        compass = data['imu'][1][-1]
        imu = [data['imu'][1][0], data['imu'][1][1], data['imu'][1][2], data['imu'][1][3]]
        obstacle_data = data['obstacle']
        obstacle_distance = -1

        if obstacle_data[0] > -1:
            obstacle_distance = obstacle_data[1][0]
            actor = obstacle_data[1][1]
            if 'static' in actor.type_id:
                obstacle_distance=-1
            else:
                ego_vehicle_point = self._map.get_waypoint(self._vehicle.get_location(),
                                                           project_to_road=False)
                ego_vehicle_road_point = self._map.get_waypoint(self._vehicle.get_location(),
                                                                lane_type=carla.LaneType.Driving, project_to_road=True)
                if ego_vehicle_point:
                    ego_vehicle_lane_id = ego_vehicle_point.lane_id
                    ego_vehicle_road_id = ego_vehicle_road_point.road_id
                else:
                    ego_vehicle_lane_id = -100
                    ego_vehicle_road_id = -100

                other_vehicle_point = self._map.get_waypoint(actor.get_location(), project_to_road=False)
                other_vehicle_road_point = self._map.get_waypoint(actor.get_location(),
                                                                  lane_type=carla.LaneType.Driving, project_to_road=True)
                if other_vehicle_point:
                    other_vehicle_lane_id = other_vehicle_point.lane_id
                    other_vehicle_road_id = other_vehicle_road_point.road_id
                else:
                    other_vehicle_lane_id = -101
                    other_vehicle_road_id = -101

                if ego_vehicle_lane_id != other_vehicle_lane_id and ego_vehicle_road_id == other_vehicle_road_id:
                    obstacle_distance = -1
                else:
                    transforms = CarlaDataProvider.get_transform(actor)
                    actor_speed = CarlaDataProvider.get_velocity(actor)
                    vehicle_theta = abs(transforms.rotation.yaw - data['imu'][1][3])
                    if vehicle_theta > 180:
                        vehicle_theta = 360 - vehicle_theta
                    if vehicle_theta > 90 and actor_speed < 0.01 and 'vehicle' in actor.type_id:
                        obstacle_distance = -1

        data = {
            'last_rgb': rgb,
            'full_gps': full_gps,
            'gps': gps,
            'lidar': lidar,
            'speed': speed,
            'compass': compass,
            'imu': imu,
            'target_diff': target_diff,
            'topdown_seg': None,
            'obstacle': obstacle_distance,
            'new_event_list': new_event_list
        }

        return data
    def _tick_scenario(self):
        world = CarlaDataProvider.get_world()
        if world:
            snapshot = world.get_snapshot()
            if snapshot:
                timestamp = snapshot.timestamp
            else:
                return False
        else:
            return False

        if self._timestamp_last_run < timestamp.elapsed_seconds:
            self._timestamp_last_run = timestamp.elapsed_seconds
            GameTime.on_carla_tick(timestamp)
            CarlaDataProvider.on_carla_tick()
            return True

        return False

    def get_theta(self, cur_node, pre_node, pos, tick_data, route_len, pixels_per_meter=3.66):
        if cur_node is None:
            return 0, 0
        size_x = 144
        size_y = 256
        compass = tick_data['compass']
        compass = 0.0 if np.isnan(compass) else compass
        compass = compass + np.pi / 2
        R = np.array([
            [np.cos(compass), -np.sin(compass)],
            [np.sin(compass), np.cos(compass)],
        ])
        x1, y1 = pixels_per_meter * (R.T.dot(pre_node - pos))
        x1 += size_x / 2
        y1 += size_y / 2
        x2, y2 = pixels_per_meter * (R.T.dot(cur_node - pos))
        x2 += size_x / 2
        y2 += size_y / 2

        x0, y0 = size_x / 2, size_y / 2

        # version-3
        location = carla.Vector3D(tick_data['full_gps'][0], tick_data['full_gps'][1], tick_data['full_gps'][2])
        tail_close_pt = self.rotate_point(carla.Vector3D(0.0001, 0.0, tick_data['full_gps'][2]),
                                          tick_data['imu'][3] - 90)

        tail_close_pt = location + tail_close_pt
        tail_close_pt = [tail_close_pt.x, tail_close_pt.y]

        tail_close_pt = (tail_close_pt - self._waypoint_planner.mean) * self._waypoint_planner.scale

        head_close_pt = self.rotate_point(carla.Vector3D(-0.000025, 0.0, tick_data['full_gps'][2]),
                                          tick_data['imu'][3] - 90)
        head_close_pt = self.normalize(head_close_pt, 0.000025)
        head_close_pt = location + head_close_pt
        head_close_pt = [head_close_pt.x, head_close_pt.y]
        head_close_pt = (head_close_pt - self._waypoint_planner.mean) * self._waypoint_planner.scale
        distance = self.compute_distance(pre_node, head_close_pt)
        # if DEBUG:
        #     self.debug.clear()
        #
        #     # self.debug.line(pos, pre_node, cur_node)
        #     self.debug.line(pos, pos, tail_close_pt, color=(255, 0, 0))
        #     self.debug.line(pos, pos, head_close_pt, color=(0, 255, 0))
        #     self.debug.dot(pos, pre_node, color=(0, 0, 255), r=5)
        #     self.debug.show()

        pre_pos = tail_close_pt
        x3, y3 = pixels_per_meter * (R.T.dot(pre_pos - pos))
        x3 += size_x / 2
        y3 += size_y / 2
        a = [x2 - x1, y2 - y1]  # route vector
        b = [x0 - x3, y0 - y3]  # vehicle vector
        location = [location.x, location.y]
        gps_location = (location - self._waypoint_planner.mean) * self._waypoint_planner.scale
        vector1 = gps_location - tail_close_pt
        # if vector1[1] < 0:
        #     vector1 = -vector1
        vector2 = cur_node - gps_location
        x4, y4 = pixels_per_meter * (R.T.dot(vector2))
        if math.sqrt(a[0] * a[0] + a[1] * a[1]) < 1e-3 or math.sqrt(b[0] * b[0] + b[1] * b[1]) < 1e-3:
            theta = self.pre_theta
        else:
            # theta = math.atan2(vector2[1], vector2[0]) - math.atan2(vector1[1], vector1[0])

            theta = (vector1[0] * vector2[0] + vector1[1] * vector2[1]) / (
                    math.sqrt(vector1[0] ** 2 + vector1[1] ** 2) * math.sqrt(vector2[0] ** 2 + vector2[1] ** 2))
            theta = max(theta, -1)
            theta = min(theta, 1)
            theta = np.arccos(theta)
            if route_len == 2 and y4 > 0:
                theta = np.pi - theta
        self.pre_theta = theta
        if distance < 0.5:
            distance = 0
        if np.isnan(theta):
            return 0, distance
        return theta, distance

    def compute_distance(self, node1, node2):
        return np.sqrt((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2)


    def normalize(self, point, length):
        length = abs(length)
        len = math.sqrt(point.x ** 2 + point.y ** 2)
        point.x = point.x * length / len
        point.y = point.y * length / len
        return point

    def rotate_point(self, point, angle):
        """
        rotate a given point by a given angle
        """
        x_ = math.cos(math.radians(angle)) * point.x - math.sin(math.radians(angle)) * point.y
        y_ = math.sin(math.radians(angle)) * point.x + math.cos(math.radians(angle)) * point.y
        return carla.Vector3D(x_, y_, point.z)


    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        ds_ids = downsample_route(global_plan_world_coord, 50)
        self._global_plan_world_coord = [(global_plan_world_coord[x][0], global_plan_world_coord[x][1]) for x in ds_ids]
        self._global_plan = [global_plan_gps[x] for x in ds_ids]
        self._plan_HACK = global_plan_world_coord
        self._plan_gps_HACK = global_plan_gps

        self._waypoint_planner = RoutePlanner(4.0, 50)
        self._waypoint_planner.set_route(self._plan_gps_HACK, True)


    def _preprocess_sensor_spec(self, sensor_spec):
        type_ = sensor_spec["type"]
        id_ = sensor_spec["id"]
        attributes = {}

        if type_ == 'sensor.opendrive_map':
            attributes['reading_frequency'] = sensor_spec['reading_frequency']
            sensor_location = carla.Location()
            sensor_rotation = carla.Rotation()

        elif type_ == 'sensor.speedometer':
            delta_time = CarlaDataProvider.get_world().get_settings().fixed_delta_seconds
            attributes['reading_frequency'] = 1 / delta_time
            sensor_location = carla.Location()
            sensor_rotation = carla.Rotation()

        if type_ == 'sensor.camera.rgb':
            attributes['image_size_x'] = str(sensor_spec['width'])
            attributes['image_size_y'] = str(sensor_spec['height'])
            attributes['fov'] = str(sensor_spec['fov'])

            sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                             z=sensor_spec['z'])
            sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                             roll=sensor_spec['roll'],
                                             yaw=sensor_spec['yaw'])

        elif type_ == 'sensor.lidar.ray_cast':
            attributes['range'] = str(85)
            attributes['rotation_frequency'] = str(10)
            attributes['channels'] = str(64)
            attributes['upper_fov'] = str(10)
            attributes['lower_fov'] = str(-30)
            attributes['points_per_second'] = str(600000)
            attributes['atmosphere_attenuation_rate'] = str(0.004)
            attributes['dropoff_general_rate'] = str(0.45)
            attributes['dropoff_intensity_limit'] = str(0.8)
            attributes['dropoff_zero_intensity'] = str(0.4)

            sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                             z=sensor_spec['z'])
            sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                             roll=sensor_spec['roll'],
                                             yaw=sensor_spec['yaw'])

        elif type_ == 'sensor.other.radar':
            attributes['horizontal_fov'] = str(sensor_spec['horizontal_fov'])  # degrees
            attributes['vertical_fov'] = str(sensor_spec['vertical_fov'])  # degrees
            attributes['points_per_second'] = '1500'
            attributes['range'] = '100'  # meters

            sensor_location = carla.Location(x=sensor_spec['x'],
                                             y=sensor_spec['y'],
                                             z=sensor_spec['z'])
            sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                             roll=sensor_spec['roll'],
                                             yaw=sensor_spec['yaw'])

        elif type_ == 'sensor.other.gnss':
            attributes['noise_alt_stddev'] = str(0.000005)
            attributes['noise_lat_stddev'] = str(0.000005)
            attributes['noise_lon_stddev'] = str(0.000005)
            attributes['noise_alt_bias'] = str(0.0)
            attributes['noise_lat_bias'] = str(0.0)
            attributes['noise_lon_bias'] = str(0.0)

            sensor_location = carla.Location(x=sensor_spec['x'],
                                             y=sensor_spec['y'],
                                             z=sensor_spec['z'])
            sensor_rotation = carla.Rotation()

        elif type_ == 'sensor.other.imu':
            attributes['noise_accel_stddev_x'] = str(0.001)
            attributes['noise_accel_stddev_y'] = str(0.001)
            attributes['noise_accel_stddev_z'] = str(0.015)
            attributes['noise_gyro_stddev_x'] = str(0.001)
            attributes['noise_gyro_stddev_y'] = str(0.001)
            attributes['noise_gyro_stddev_z'] = str(0.001)

            sensor_location = carla.Location(x=sensor_spec['x'],
                                             y=sensor_spec['y'],
                                             z=sensor_spec['z'])
            sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                             roll=sensor_spec['roll'],
                                             yaw=sensor_spec['yaw'])

        elif type_=='sensor.other.obstacle':
            sensor_location = carla.Location()
            sensor_rotation = carla.Rotation()
        sensor_transform = carla.Transform(sensor_location, sensor_rotation)

        return type_, id_, sensor_transform, attributes
    def setup_sensors(self, sensors, debug_mode=False):
        """
        Create the sensors defined by the user and attach them to the ego-vehicle
        :param vehicle: ego vehicle
        :return:
        """
        vehicle = self.ego_vehicles[0]
        bp_library = CarlaDataProvider.get_world().get_blueprint_library()
        for sensor_spec in sensors:
            # These are the pseudosensors (not spawned)
            type_, id_, sensor_transform, attributes = self._preprocess_sensor_spec(sensor_spec)

            # These are the pseudosensors (not spawned)
            if type_ == 'sensor.opendrive_map':
                sensor = OpenDriveMapReader(vehicle, attributes['reading_frequency'])
            elif type_ == 'sensor.speedometer':
                sensor = SpeedometerReader(vehicle, attributes['reading_frequency'])

            # These are the sensors spawned on the carla world
            else:
                bp = bp_library.find(type_)
                for key, value in attributes.items():
                    bp.set_attribute(str(key), str(value))
                sensor = CarlaDataProvider.get_world().spawn_actor(bp, sensor_transform, vehicle)
            # setup callback
            sensor.listen(CallBack(id_, type_, sensor, self.sensor_interface))
            self._sensors_list.append(sensor)

        # Tick once to spawn the sensors
        for _ in range(30):
            CarlaDataProvider.get_world().tick()

    @staticmethod
    def _get_tcp_port(port: int = 0):
        """Get a free TCP port number.

        Args:
          port (Optional[int]): Port number. When set to `0`, it will be assigned a free port dynamically.

        Returns:
            A port number requested if free, otherwise an unhandled exception would be thrown.
        """
        s = socket.socket()
        s.bind(("", port))
        server_port = s.getsockname()[1]
        s.close()
        return server_port


    def _init_server(self):
        """Initialize CARLA server and client."""
        #TODO:检测host若不为local则远程开启
        logging.debug('Initializing new Carla server...')
        if not self.fixed_port:
            self.port = self._get_tcp_port()
        main_thread_execution = True
        logging.debug(f"Port: {self.port}\n")
        if not self.render:
            self._server_process = subprocess.Popen(
                (   f'bash "{SERVER_BINARY}" -RenderOffScreen -graphicsadaper=1'
                    f" -carla-rpc-port={self.port} "
                    f" -quality-level={self.quality_level} -Resx=600 -Resy=480"
                ),
                shell=True,
                preexec_fn=os.setsid
            )
        else:
            self._server_process = subprocess.Popen(
                (  f'SDL_HINT_CUDA_DEVICE={self.cuda_id} '
                    f'bash "{SERVER_BINARY}"'
                    f" -world-port={self.port}"
                    f" -quality-level={self.quality_level} -Resx=600 -Resy=480"
                ),
                shell=True,
                preexec_fn=os.setsid
            )
        time.sleep(5)

if __name__=="__main__":
    # subprocess.Popen(
    #     "killall -9 -r CarlaUE4",
    #     shell=True,
    #     preexec_fn=os.setsid
    # )

    plt.ion()
    plt.figure()
    all_config = Config.fromfile("config/agent_config.py")
    env_cfg = all_config.env_cfg
    rank = 0
    env_cfg.rank = rank
    env_cfg.port = env_cfg.port[rank]
    env_cfg.routes = env_cfg.routes[rank]
    env_cfg.scenarios = env_cfg.scenarios[rank]
    env_cfg.town = env_cfg.town[rank]
    env_cfg.seq_length = all_config.rollout_cfg.seq_length
    env=EnvWrapper(env_cfg)
    env.reset()
    while True:
        tick_data, rewards, done, info=env.step([0,0.5,0])
        if done:
            env.reset()