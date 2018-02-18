from serpent.game_agent import GameAgent

from serpent.game_frame import GameFrame
from serpent.frame_grabber import FrameGrabber

from serpent.input_controller import KeyboardKey

from serpent.config import config

from datetime import datetime

import skimage.io
import skimage.transform
import skimage.measure
import skimage.util

import numpy as np

import random
import time
import collections
import subprocess
import shlex
import os

from .helpers.helper import expand_bounding_box
from .helpers.terminal_printer import TerminalPrinter

from .helpers.ppo import SerpentPPO


class SerpentSuperflightGameAgent(GameAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play
        self.frame_handlers["RANDOM"] = self.handle_random

        self.frame_handler_setups["PLAY"] = self.setup_play
        self.frame_handler_setups["RANDOM"] = self.setup_random

        self.frame_handler_pause_callbacks["PLAY"] = self.handle_play_pause
        self.frame_handler_pause_callbacks["RANDOM"] = self.handle_random_pause

        self.reward_functions["AEROBATICS"] = self.reward_superflight_simple

        self.analytics_client = None

        self.game_inputs = {
            "DESCEND": [KeyboardKey.KEY_W],
            "DESCEND-RIGHT": [KeyboardKey.KEY_W, KeyboardKey.KEY_D],
            "RIGHT": [KeyboardKey.KEY_D],
            "ASCEND-RIGHT": [KeyboardKey.KEY_S, KeyboardKey.KEY_D],
            "ASCEND": [KeyboardKey.KEY_S],
            "ASCEND-LEFT": [KeyboardKey.KEY_S, KeyboardKey.KEY_A],
            "LEFT": [KeyboardKey.KEY_A],
            "DESCEND-LEFT": [KeyboardKey.KEY_W, KeyboardKey.KEY_A],
            "NOOP": []
        }

        self.printer = TerminalPrinter()

    def relaunch(self):
        self.printer.flush()

        self.printer.add("")
        self.printer.add("This game has a NASTY memory leak! Periodic restarts are a must.")
        self.printer.add("")

        self.printer.add("Hot-swapping the game window the agent is looking at...")
        self.printer.add("The experiment will resume once the new game window is ready!")
        self.printer.add("")

        self.printer.flush()

        self.game.stop_frame_grabber()

        time.sleep(1)

        self.input_controller.tap_keys([
            KeyboardKey.KEY_LEFT_ALT,
            KeyboardKey.KEY_F4
        ])

        time.sleep(1)

        subprocess.call(shlex.split("serpent launch Superflight"))
        self.game.launch(dry_run=True)

        self.game.start_frame_grabber()
        self.game.redis_client.delete(config["frame_grabber"]["redis_key"])

        while self.game.redis_client.llen(config["frame_grabber"]["redis_key"]) == 0:
            time.sleep(0.1)

        self.game.window_controller.focus_window(self.game.window_id)

        time.sleep(3)
        self.printer.flush()

    def setup_play(self):
        self.run_count = 0
        self.run_reward = 0

        self.observation_count = 0

        self.delay_fuzzing_durations = [0.05, 0.1, 0.2]
        self.delay_fuzzing_observation_cap = 100000

        self.performed_inputs = collections.deque(list(), maxlen=8)

        self.reward_10 = collections.deque(list(), maxlen=10)
        self.reward_100 = collections.deque(list(), maxlen=100)
        self.reward_1000 = collections.deque(list(), maxlen=1000)

        self.average_reward_10 = 0
        self.average_reward_100 = 0
        self.average_reward_1000 = 0

        self.score_10 = collections.deque(list(), maxlen=10)
        self.score_100 = collections.deque(list(), maxlen=100)
        self.score_1000 = collections.deque(list(), maxlen=1000)

        self.average_score_10 = 0
        self.average_score_100 = 0
        self.average_score_1000 = 0

        self.top_score = 0
        self.top_score_run = 0

        self.previous_score = 0

        # Measured on the Serpent.AI Lab Stream (Feb 12 2018)
        self.random_average_score = 67.51
        self.random_top_score = 5351
        self.random_runs = 2700

        self.death_check = False
        self.just_relaunched = False

        self.frame_buffer = None

        self.ppo_agent = SerpentPPO(
            frame_shape=(100, 100, 4),
            game_inputs=self.game_inputs
        )

        # Warm Agent?
        game_frame_buffer = FrameGrabber.get_frames([0, 2, 4, 6], frame_type="PIPELINE")
        self.ppo_agent.generate_action(game_frame_buffer)

        self.started_at = datetime.utcnow().isoformat()

    def setup_random(self):
        self.run_count = 0

        self.performed_inputs = collections.deque(list(), maxlen=8)

        self.top_score = 0
        self.average_score = 0
        self.previous_score = 0

        self.death_check = False
        self.just_relaunched = False

    def handle_play(self, game_frame):
        # Game crash detection
        game_frame_buffer = FrameGrabber.get_frames([0, 60])
        frame_1, frame_2 = game_frame_buffer.frames

        if np.array_equal(frame_1, frame_2):
            self.printer.add("")
            self.printer.add("Game appears to have crashed... Relaunching!")
            self.printer.flush()

            self.relaunch()
            self.just_relaunched = True

            self.frame_buffer = None

            return None

        # Check for recent game relaunch
        if self.just_relaunched:
            self.just_relaunched = False
            self.frame_buffer = None

            self.input_controller.tap_key(KeyboardKey.KEY_ENTER)

            self.printer.flush()
            return None

        self.printer.add("")
        self.printer.add("Serpent.AI Lab - Superflight")
        self.printer.add("Stage 2: Reinforcement Learning: Training a PPO Agent")
        self.printer.add("")
        self.printer.add(f"Stage Started At: {self.started_at}")
        self.printer.add(f"Current Run: #{self.run_count}")
        self.printer.add("")

        reward = self.reward_superflight_simple([None, None, game_frame, None])

        if self.frame_buffer is not None:
            if reward == 0 and self.death_check is False:
                pass
            else:
                self.ppo_agent.observe(reward, terminal=(reward == 0))
                self.observation_count += 1

        if reward > 0:
            self.death_check = False
            self.run_reward += reward

            self.frame_buffer = FrameGrabber.get_frames([0, 2, 4, 6], frame_type="PIPELINE")

            action, label, game_input = self.ppo_agent.generate_action(self.frame_buffer)

            self.performed_inputs.appendleft(label)
            self.input_controller.handle_keys(game_input)

            self.printer.add(f"Current Reward: {reward}")
            self.printer.add(f"Run Reward: {self.run_reward}")
            self.printer.add("")


            if self.observation_count < self.delay_fuzzing_observation_cap:
                self.printer.add(f"Observation Count: {self.observation_count}")
                self.printer.add("")

            self.printer.add(f"Average Rewards (Last 10 Runs): {self.average_reward_10}")
            self.printer.add(f"Average Rewards (Last 100 Runs): {self.average_reward_100}")
            self.printer.add(f"Average Rewards (Last 1000 Runs): {self.average_reward_1000}")
            self.printer.add("")
            self.printer.add(f"Previous Run Score: {self.previous_score}")
            self.printer.add("")
            self.printer.add(f"Average Score (Last 10 Runs): {self.average_score_10}")
            self.printer.add(f"Average Score (Last 100 Runs): {self.average_score_100}")
            self.printer.add(f"Average Score (Last 1000 Runs): {self.average_score_1000}")
            self.printer.add("")
            self.printer.add(f"Top Score: {self.top_score} (Run #{self.top_score_run})")
            self.printer.add("")
            self.printer.add(f"Random Agent Average Score: {self.random_average_score} (over {self.random_runs} runs)")
            self.printer.add(f"Random Agent Top Score: {self.random_top_score}")
            self.printer.add("")
            self.printer.add("Latest Inputs:")
            self.printer.add("")

            for i in self.performed_inputs:
                self.printer.add(i)

            self.printer.flush()

            if self.observation_count < self.delay_fuzzing_observation_cap and np.random.uniform(0, 1) > 0.5:
                time.sleep(random.choice(self.delay_fuzzing_durations))
        else:
            if not self.death_check:
                self.death_check = True

                self.printer.flush()
                return None
            else:
                self.input_controller.handle_keys([])

                self.run_count += 1
                self.performed_inputs.clear()

                self.reward_10.appendleft(self.run_reward)
                self.reward_100.appendleft(self.run_reward)
                self.reward_1000.appendleft(self.run_reward)

                self.run_reward = 0

                self.average_reward_10 = float(np.mean(self.reward_10))
                self.average_reward_100 = float(np.mean(self.reward_100))
                self.average_reward_1000 = float(np.mean(self.reward_1000))

                score = self.game.api.parse_score(game_frame)
                self.previous_score = score

                self.printer.add(f"The game agent just died with score: {score}")

                self.score_10.appendleft(score)
                self.score_100.appendleft(score)
                self.score_1000.appendleft(score)

                self.average_score_10 = float(np.mean(self.score_10))
                self.average_score_100 = float(np.mean(self.score_100))
                self.average_score_1000 = float(np.mean(self.score_1000))

                if score > self.top_score:
                    self.printer.add(f"NEW RECORD!")

                    self.top_score = score
                    self.top_score_run = self.run_count - 1

                self.printer.add("")

                self.frame_buffer = None

                # Memory Leak Relaunch Check
                if (time.time() - self.game.launched_at) > 3600:
                    self.relaunch()
                    self.just_relaunched = True
                else:
                    for i in range(3):
                        self.input_controller.tap_key(KeyboardKey.KEY_UP)

                    if not self.run_count % 5:
                        self.printer.add("Changing Map...")
                        self.printer.flush()

                        self.game.api.change_map(input_controller=self.input_controller)
                    else:
                        self.printer.add("Restarting...")
                        self.printer.flush()

                        self.input_controller.tap_key(KeyboardKey.KEY_ENTER)

                time.sleep(0.5)

    def handle_random(self, game_frame):
        # Game Crash Detection
        game_frame_buffer = FrameGrabber.get_frames([0, 60])
        frame_1, frame_2 = game_frame_buffer.frames

        if np.array_equal(frame_1, frame_2):
            self.printer.add("")
            self.printer.add("Game appears to have crashed... Relaunching!")
            self.printer.flush()

            self.relaunch()
            self.just_relaunched = True

            return None

        self.printer.add("")
        self.printer.add("Serpent.AI Lab - Superflight")
        self.printer.add("Stage 1: Collecting Random Agent Data...")
        self.printer.add("")
        self.printer.add(f"Current Run: #{self.run_count}")
        self.printer.add("")

        if self.just_relaunched:
            self.just_relaunched = False
            self.input_controller.tap_key(KeyboardKey.KEY_ENTER)

            self.printer.flush()
            return None

        reward = self.reward_superflight_simple([None, None, game_frame, None])

        if reward > 0:
            self.death_check = False

            game_input_key = random.choice(list(self.game_inputs.keys()))
            self.performed_inputs.appendleft(game_input_key)

            self.input_controller.handle_keys(self.game_inputs[game_input_key])

            self.printer.add(f"Average Score: {self.average_score}")
            self.printer.add(f"Top Score: {self.top_score}")
            self.printer.add("")
            self.printer.add(f"Previous Run Score: {self.previous_score}")
            self.printer.add("")
            self.printer.add("")
            self.printer.add(f"Reward: {reward}")
            self.printer.add("")
            self.printer.add("")
            self.printer.add("Latest Inputs:")
            self.printer.add("")

            for i in self.performed_inputs:
                self.printer.add(i)

            self.printer.flush()
        else:
            if not self.death_check:
                self.death_check = True

                self.printer.flush()
                return None
            else:
                self.input_controller.handle_keys([])

                self.run_count += 1
                self.performed_inputs.clear()

                score = self.game.api.parse_score(game_frame)

                self.previous_score = score
                self.average_score += ((score - self.average_score) / self.run_count)

                self.printer.add(f"The game agent just died with score: {score}")

                if score > self.top_score:
                    self.printer.add(f"NEW RECORD!")
                    self.top_score = score

                self.printer.add("")

                # Memory Leak Relaunch Check
                if (time.time() - self.game.launched_at) > 3600:
                    self.relaunch()
                    self.just_relaunched = True
                else:
                    for i in range(3):
                        self.input_controller.tap_key(KeyboardKey.KEY_UP)

                    if not self.run_count % 5:
                        self.printer.add("Changing Map...")
                        self.printer.flush()

                        self.game.api.change_map(input_controller=self.input_controller)
                    else:
                        self.printer.add("Restarting...")
                        self.printer.flush()

                        self.input_controller.tap_key(KeyboardKey.KEY_ENTER)

                time.sleep(0.5)

    def handle_play_pause(self):
        self.ppo_agent.agent.save_model(directory=os.path.join(os.getcwd(), "datasets", "ppo_model"))

    def handle_random_pause(self):
        pass

    def reward_superflight_simple(self, frames, **kwargs):
        bw_frame = self._reward_superflight_simple_preprocess(frames)

        is_scoring = bw_frame.max() == 255
        white_pixel_count = bw_frame[bw_frame == 255].size

        if not self.game.api.is_alive(GameFrame(frames[-2].frame), self.sprite_identifier):
            return 0

        if white_pixel_count > 20000:
            return 0.5

        if is_scoring:
            return 0.5 + ((white_pixel_count / 20000) / 2)
        else:
            return np.random.uniform(0.1, 0.3)  # Returning a static value here causes early convergence. Fuzzing!

    def _reward_superflight_simple_preprocess(self, frames):
        bw_frame = skimage.util.img_as_ubyte(np.all(frames[-2].frame > 240, axis=-1))

        label_frame = skimage.measure.label(bw_frame)
        regions = skimage.measure.regionprops(label_frame)

        clean_frame = np.zeros(bw_frame.shape, dtype="uint8")

        bounding_boxes = list()

        for region in regions:
            if region.area <= 10 or region.area > 1000:
                continue

            y0, y1, x0, x1 = expand_bounding_box(region.bbox, bw_frame.shape, 5, 5)
            aspect_ratio = (x1 - x0) / (y1 - y0)

            if aspect_ratio < 0.4 or aspect_ratio > 1.0:
                continue

            bounding_boxes.append([y0, y1, x0, x1])

        for b in bounding_boxes:
            for bb in bounding_boxes:
                if b == bb:
                    continue

                if b[0] in range(bb[0], bb[1] + 1) or b[1] in range(bb[0], bb[1] + 1):
                    if b[2] in range(bb[2], bb[3] + 1) or b[3] in range(bb[2], bb[3] + 1):
                        clean_frame[b[0]:b[1], b[2]:b[3]] = 255
                        break

        return clean_frame

