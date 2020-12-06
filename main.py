from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
sys.path.append(r'C:\Users\dell\Anaconda3\envs\pytorch\Lib\site-packages')
sys.path.append(r'C:\Users\dell\Anaconda3\Lib\site-packages')
import time
import importlib
import threading

from absl import app
from absl import flags
import pysc2
from pysc2 import maps
from pysc2.env import available_actions_printer
from pysc2.env import sc2_env
from pysc2.lib import stopwatch
from pysc2.lib import point_flag
import tensorflow as tf

from run_loop import run_loop

COUNTER = 0
LOCK = threading.Lock()
FLAGS = flags.FLAGS

# About training, sc2 independent
flags.DEFINE_bool("training", True, "Whether to train agents.")
flags.DEFINE_bool("teaching", False, "Use teaching scripts to training agent.")
flags.DEFINE_bool("continuation", False, "Continuously training.")
flags.DEFINE_float("learning_rate", 5e-4, "Learning rate for training.")
flags.DEFINE_float("discount", 0.99, "Discount rate for future rewards.")
flags.DEFINE_integer("max_steps", int(1e5), "Total steps for training.")
flags.DEFINE_integer("snapshot_step", int(1e3), "Step for snapshot.")
flags.DEFINE_string("snapshot_path", "./snapshot/", "Path for snapshot.")
flags.DEFINE_string("log_path", "./log/", "Path for log.")
flags.DEFINE_string("device", "0", "Device for training.")
flags.DEFINE_integer("evaluate_every", 20, "Evaluate every how many training episodes.")

# Agent structure
flags.DEFINE_string("agent", "agents.a3c_agent.A3CAgent", "Which agent to run.")
flags.DEFINE_string("net", "fcn", "atari or fcn.")

# Env setup
flags.DEFINE_string("map", "MoveToBeacon", "Name of a map to use.")

# Resolution
# flags.DEFINE_integer("screen_resolution", 64, "Resolution for screen feature layers.")
# flags.DEFINE_integer("minimap_resolution", 64, "Resolution for minimap feature layers.")
point_flag.DEFINE_point("feature_screen_size", "64",
                        "Resolution for screen feature layers.")
point_flag.DEFINE_point("feature_minimap_size", "64",
                        "Resolution for minimap feature layers.")
point_flag.DEFINE_point("rgb_screen_size", None,
                        "Resolution for rendered screen.")
point_flag.DEFINE_point("rgb_minimap_size", None,
                        "Resolution for rendered minimap.")
flags.DEFINE_enum("action_space", None, sc2_env.ActionSpace._member_names_,  # pylint: disable=protected-access
                  "Which action space to use. Needed if you take both feature "
                  "and rgb observations.")

# Agent-stepping settings
flags.DEFINE_integer("max_agent_steps", 3200, "Total agent steps.")
flags.DEFINE_integer("game_steps_per_episode", None, "Game steps per episode.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")
flags.DEFINE_string("agent_name", None,
                    "Name of the agent in replays. Defaults to the class name.")
flags.DEFINE_enum("agent_race", "random", sc2_env.Race._member_names_, "Agent's race.")
flags.DEFINE_enum("bot_race", None, sc2_env.Race._member_names_, "Bot's race.")
flags.DEFINE_enum("difficulty", None, sc2_env.Difficulty._member_names_, "Bot's strength.")

# Running settings
flags.DEFINE_bool("render", False, "Whether to render with pygame.") # only render ONE process guaranteed
flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")
flags.DEFINE_integer("parallel", 1, "How many instances to run in parallel.")
flags.DEFINE_bool("save_replay", False, "Whether to save a replay at the end.")

FLAGS(sys.argv)
if FLAGS.training:
  PARALLEL = FLAGS.parallel
  MAX_AGENT_STEPS = FLAGS.max_agent_steps
  DEVICE = ['/gpu:'+dev for dev in FLAGS.device.split(',')]
else:
  PARALLEL = 1
  MAX_AGENT_STEPS = 1e5
  # DEVICE = ['/cpu:0']
  DEVICE = ['/gpu:'+dev for dev in FLAGS.device.split(',')]

LOG = FLAGS.log_path+FLAGS.map+'/'+FLAGS.net
SNAPSHOT = FLAGS.snapshot_path+FLAGS.map+'/'+FLAGS.net
SCORE_LOG = FLAGS.log_path+'score_log/'+FLAGS.map+'/'+FLAGS.net
if not os.path.exists(LOG):
  os.makedirs(LOG)
if not os.path.exists(SNAPSHOT):
  os.makedirs(SNAPSHOT)
if not os.path.exists(SCORE_LOG):
  os.makedirs(SCORE_LOG)


def evaluate_k_episodes_and_avg(agent, env, k=5, write_results=False):
  # Evaluate multiple episodes and compute scores
  agent.training = True
  avg_score, max_score, min_score = 0, -1, 100
  print("Evaluating...")
  for epi in range(k):
    print("Running episode {}/{}.".format(epi, k))
    for recorder, is_done in run_loop([agent], env, MAX_AGENT_STEPS):
      if not is_done:
        continue
      obs = recorder[0].observation
      score = obs["score_cumulative"].score
      avg_score += score
      max_score = max(max_score, score)
      min_score = min(min_score, score)
      break
  
  avg_score /= k
  print("Max/Min/Avg score: {} / {} / {}".format(max_score, min_score, avg_score))
  agent.training = True

  # write results to score_log file for visualization
  if write_results:
    log_fn = SCORE_LOG + '/log.dat'
    with open(log_fn, 'a') as fout:
      fout.write('\t'.join(map(str, [COUNTER, max_score, min_score, avg_score])) + '\n')

  return avg_score


def train_one_episode(agent, env):
  # Step & update weights in one episode
  # Only for a single player!
  global COUNTER
  replay_buffer = []
  for recorder, is_done in run_loop([agent], env, MAX_AGENT_STEPS, COUNTER):
    replay_buffer.append(recorder)
    if not is_done:
      continue
    counter = 0
    with LOCK:
      COUNTER += 1
      counter = COUNTER
    # Learning rate schedule
    learning_rate = FLAGS.learning_rate * (1 - 0.9 * counter / FLAGS.max_steps)
    agent.update(replay_buffer, FLAGS.discount, learning_rate, counter)
    break


def run_train_and_eval(agent, players, map_name, visualize):
  global COUNTER
  with sc2_env.SC2Env(
    map_name=map_name,
    players=players,
    step_mul=FLAGS.step_mul,
    #screen_size_px=(FLAGS.screen_resolution, FLAGS.screen_resolution),
    #minimap_size_px=(FLAGS.minimap_resolution, FLAGS.minimap_resolution),
    agent_interface_format=sc2_env.parse_agent_interface_format(
          feature_screen=FLAGS.feature_screen_size,
          feature_minimap=FLAGS.feature_minimap_size,
          rgb_screen=FLAGS.rgb_screen_size,
          rgb_minimap=FLAGS.rgb_minimap_size,
          action_space=FLAGS.action_space),
    game_steps_per_episode=FLAGS.game_steps_per_episode,
    visualize=visualize) as env:

    env = available_actions_printer.AvailableActionsPrinter(env)

    if not FLAGS.training:
      evaluate_k_episodes_and_avg(agent, env, k=10)
      exit(0)

    max_avg_score = 0.
    while True:
      train_one_episode(agent, env)
      if COUNTER % FLAGS.evaluate_every == 1 and COUNTER >= 50:
        avg_sc = evaluate_k_episodes_and_avg(agent, env, k=6, write_results=True)
        if avg_sc > max_avg_score:
          max_avg_score = avg_sc
          agent.save_model(SNAPSHOT, COUNTER)
      print("Current max average score: {}".format(max_avg_score))
    
    if FLAGS.save_replay:
      env.save_replay(agent.name)


def run_thread(agent, players, map_name, visualize):
  global COUNTER
  with sc2_env.SC2Env(
    map_name=map_name,
    players=players,
    step_mul=FLAGS.step_mul,
    #screen_size_px=(FLAGS.screen_resolution, FLAGS.screen_resolution),
    #minimap_size_px=(FLAGS.minimap_resolution, FLAGS.minimap_resolution),
    agent_interface_format=sc2_env.parse_agent_interface_format(
          feature_screen=FLAGS.feature_screen_size,
          feature_minimap=FLAGS.feature_minimap_size,
          rgb_screen=FLAGS.rgb_screen_size,
          rgb_minimap=FLAGS.rgb_minimap_size,
          action_space=FLAGS.action_space),
    game_steps_per_episode=FLAGS.game_steps_per_episode,
    visualize=visualize) as env:
    
    env = available_actions_printer.AvailableActionsPrinter(env)
    max_avg_score = 0.
    # pure evaluation
    if not FLAGS.training:
      evaluate_k_episodes_and_avg(agent, env, k=20)
      exit(0)

    # Only for a single player!
    replay_buffer = []
    for recorder, is_done in run_loop([agent], env, MAX_AGENT_STEPS, COUNTER):
      if FLAGS.training:
        replay_buffer.append(recorder)
        if is_done:
          counter = 0
          with LOCK:
            COUNTER += 1
            counter = COUNTER
          # Learning rate schedule
          learning_rate = FLAGS.learning_rate * (1 - 0.9 * counter / FLAGS.max_steps)
          agent.update(replay_buffer, FLAGS.discount, learning_rate, counter)
          replay_buffer = []
          if counter % FLAGS.snapshot_step == 1:
            agent.save_model(SNAPSHOT, counter)
          if counter >= FLAGS.max_steps:
            break
          
          # eval with interval
          if COUNTER % FLAGS.evaluate_every == 1 and COUNTER >= 0:
            avg_sc = evaluate_k_episodes_and_avg(agent, env, k=10, write_results=True)
            if avg_sc > max_avg_score:
              max_avg_score = avg_sc
              agent.save_model(SNAPSHOT, COUNTER)
          print("Current max average score: {}".format(max_avg_score))
            
      elif is_done:
        obs = recorder[-1].observation
        score = obs["score_cumulative"][0]
        print('Your score is '+str(score)+'!')
    if FLAGS.save_replay:
      env.save_replay(agent.name)


def _main(unused_argv):
  """Run agents"""
  if FLAGS.trace:
    stopwatch.sw.trace()
  elif FLAGS.profile:
    stopwatch.sw.enable()

  maps.get(FLAGS.map)  # Assert the map exists.

  # Setup agents
  agent_module, agent_name = FLAGS.agent.rsplit(".", 1)
  agent_cls = getattr(importlib.import_module(agent_module), agent_name)
  print(agent_module, agent_name, agent_cls)

  agents = []
  players = []
  players.append(sc2_env.Agent(sc2_env.Race[FLAGS.agent_race],
                               FLAGS.agent_name or agent_name))

  for i in range(PARALLEL):
    agent = agent_cls(FLAGS.training, FLAGS.feature_minimap_size[0], FLAGS.feature_screen_size[0])
    agent.build_model(i > 0, DEVICE[i % len(DEVICE)], FLAGS.net)
    agents.append(agent)

  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)

  summary_writer = tf.summary.FileWriter(LOG)
  for i in range(PARALLEL):
    agents[i].setup(sess, summary_writer)

  agent.initialize()
  if not FLAGS.training or FLAGS.continuation:
    global COUNTER
    COUNTER = agent.load_model(SNAPSHOT)

  # Run threads
  # wrapper_func = run_train_and_eval
  wrapper_func = run_thread
  threads = []
  for i in range(PARALLEL - 1):
    t = threading.Thread(target=wrapper_func, args=(agents[i], players, FLAGS.map, False))
    threads.append(t)
    t.daemon = True
    t.start()
    time.sleep(5)

  wrapper_func(agents[-1], players, FLAGS.map, FLAGS.render)

  for t in threads:
    t.join()

  if FLAGS.profile:
    print(stopwatch.sw)


if __name__ == "__main__":
  app.run(_main)
