import os
import sys
import time
import warnings
import argparse
import numpy as np
import shutil
import json

import grid2op
from grid2op.dtypes import dt_int
from grid2op.utils import ScoreL2RPN2022
from grid2op.Chronics import ChangeNothing
from grid2op.Agent import BaseAgent
from grid2op.Reward import BaseReward, RedispReward, L2RPNWCCI2022ScoreFun
from grid2op.Opponent import BaseOpponent
from grid2op.Episode import EpisodeReplay

try:
    from lightsim2grid import LightSimBackend
    MyBackend = LightSimBackend
except ImportError:
    from grid2op.Backend import PandaPowerBackend
    MyBackend = PandaPowerBackend
    print("INFO: the backend used in codalab is \"LightSimBackend\" which is not available on your system. "
          "Please visit \"https://github.com/BDonnot/lightsim2grid\" to install it. "
          "NB results difference between native backend (PandaPowerBackend) and LightSimBackend are "
          "below 1e-5 in most cases.")

DEBUG = True  # for our tests, then we switch to False


SUBMISSION_DIR_ERR = """
ERROR: Impossible to find a "submission" package.
Agents should be included in a "submission" directory
A module with a function "make_agent" to load the agent that will be assessed."
"""

MAKE_AGENT_ERR = """
ERROR:  We could NOT find a function name \"make_agent\"
in your \"submission\" package. "
We remind you that this function is mandatory and should have the signature:
 make_agent(environment, path_agent) -> agent 

 - The "agent" is the agent that will be tested.
 - The "environment" is a valid environment provided.
   It will NOT be updated during the scoring (no data are fed to it).
 - The "path_agent" is the path where your agent is located
"""

ENV_TEMPLATE_ERR = """
ERROR: There is no powergrid found for making the template environment. 
Or creating the template environment failed.
The agent will not be created and this will fail.
"""

MAKE_AGENT_ERR2 = """
ERROR: "make_agent" is present in your package, but can NOT be used.

We remind you that this function is mandatory and should have the signature:
 make_agent(environment, path_agent) -> agent

 - The "agent" is the agent that will be tested.
 - The "environment" is a valid environment provided.
   It will NOT be updated during the scoring (no data are fed to it).
 - The "path_agent" is the path where your agent is located
"""

BASEAGENT_ERR = """
ERROR: The "submitted_agent" provided should be a valid Agent. 
It should be of class that inherit "BaseAgent" (`from grid2op.Agent import BaseAgent`) base class
"""

INFO_CUSTOM_REWARD = """
INFO: No custom reward for the assessment of your agent will be used.
"""

REWARD_ERR = """
ERROR: The "training_reward" provided should be a class.
NOT a instance of a class
"""

REWARD_ERR2 = """
ERROR: The "training_reward" provided is invalid.
It should inherit the "grid2op.Reward.BaseReward" class
"""

INFO_CUSTOM_OTHER = """
INFO: No custom other_rewards for the assessment of your agent will be used.
"""

KEY_OVERLOAD_REWARD = """
WARNING: You provided the key "{0}" in the "other_reward" dictionnary. 
This will be replaced by the score of the competition, as stated in the rules. Your "{0}" key WILL BE erased by this operation.
"""

KEY_OVERLOAD_WARN = """
The key "{}" cannot be used as a custom reward. It is used internally to get compute your score.
It will be disabled and erased.
"""

BACKEND_WARN = """
WARNING: Could not load lightsim2grid.LightSimBackend, falling back on PandaPowerBackend
"""

STARTING_THE_EVALUATION = """Starting the evaluation of your agent on the private dataset"""
ENDING_THE_EVALUATION = """Starting the evaluation of your agent on the private dataset"""


def cli():
    DEFAULT_GIF_EPISODE = None
    DEFAULT_GIF_START = 0
    DEFAULT_GIF_END = 50
    DEFAULT_CLEANUP = False
    
    parser = argparse.ArgumentParser(description="Ingestion program")
    parser.add_argument("--input_path", required=True,
                        help="Path to the dataset")
    parser.add_argument("--output_path", required=True,
                        help="Path to the runner logs output dir")
    parser.add_argument("--program_path", required=True,
                        help="Path to the program dir")
    parser.add_argument("--submission_path", required=True,
                        help="Path to the submission dir")
    parser.add_argument("--config_in", required=True,
                        help="DoNothing json config input file")
    parser.add_argument("--gif_episode", required=False,
                        default=DEFAULT_GIF_EPISODE, type=str,
                        help="Name of the episode to generate a gif for")
    parser.add_argument("--gif_start", required=False,
                        default=DEFAULT_GIF_START, type=int,
                        help="Start step for gif generation")
    parser.add_argument("--gif_end", required=False,
                        default=DEFAULT_GIF_END, type=int,
                        help="End step for gif generation")
    parser.add_argument("--cleanup", required=False,
                        default=DEFAULT_CLEANUP, action='store_true',
                        help="Cleanup runner logs")
    parser.add_argument("--verbose", action='store_true',
                        help="Verbose runner output")
    return parser.parse_args()


def write_gif(output_dir, agent_path, episode_name, start_step, end_step):
    try:
        epr = EpisodeReplay(agent_path)
        epr.replay_episode(episode_name,
                           fps=2.0,
                           display=False,
                           gif_name=episode_name,
                           start_step=start_step,
                           end_step=end_step,
                           load_info=None,
                           gen_info=None,
                           line_info=None
                           )
        gif_genpath = os.path.join(agent_path, episode_name,
                                   episode_name + ".gif")
        gif_outpath = os.path.join(output_dir, episode_name + ".gif")
        print(gif_genpath, gif_outpath)
        if os.path.exists(gif_genpath):
            shutil.move(gif_genpath, gif_outpath)
    except Exception as exc_:
        print("Cannot create GIF export with error \n{}".format(exc_))


def main():
    args = cli()
    
    # read arguments
    input_dir = args.input_path
    output_dir = args.output_path
    program_dir = args.program_path
    submission_dir = args.submission_path
    config_file = args.config_in
    with open(config_file, "r") as f:
        config = json.load(f)

    # create output dir if not existing
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if DEBUG:
        print("input dir: {}".format(input_dir))
        print("output dir: {}".format(output_dir))
        print("program dir: {}".format(program_dir))
        print("submission dir: {}".format(submission_dir))

        print("input content", os.listdir(input_dir))
        print("output content", os.listdir(output_dir))
        print("program content", os.listdir(program_dir))
    print("Content received by codalab: {}".format(sorted(os.listdir(submission_dir))))

    submission_location = os.path.join(submission_dir, "submission")
    if not os.path.exists(submission_location):
        print(SUBMISSION_DIR_ERR)
        raise RuntimeError(SUBMISSION_DIR_ERR)

    # add proper directories to path
    sys.path.append(program_dir)
    sys.path.append(submission_dir)

    # import the agent's code
    try:
       from submission import make_agent
    except Exception as exc_:
        print(MAKE_AGENT_ERR)
        print("The error was: {}".format(exc_))
        raise RuntimeError(MAKE_AGENT_ERR) from exc_

    # create the backend
    try:
        from lightsim2grid import LightSimBackend
        backend_cls = LightSimBackend
    except :
        print(BACKEND_WARN)
        from grid2op.Backend import PandaPowerBackend
        backend_cls = PandaPowerBackend
    
    # create the template environment
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env_template = grid2op.make(input_dir,
                                        chronics_class=ChangeNothing,
                                        opponent_class=BaseOpponent,
                                        backend=backend_cls(),
                                        )
            env_template._init_env_path = submission_dir
            env_template._init_grid_path = None
            type(env_template)._PATH_ENV = submission_dir
            
    except Exception as exc_:
        print(ENV_TEMPLATE_ERR)
        print("The error was: {}".format(exc_))
        raise RuntimeError(ENV_TEMPLATE_ERR) from exc_

    # create the agent
    try:
        submitted_agent = make_agent(env_template, submission_location)
    except Exception as exc_:
        print(MAKE_AGENT_ERR2)
        print("The error was: {}".format(exc_))
        raise RuntimeError(MAKE_AGENT_ERR2) from exc_

    if not isinstance(submitted_agent, BaseAgent):
        print(BASEAGENT_ERR)
        raise RuntimeError(BASEAGENT_ERR)

    # import the rewards and other things
    try:
        from submission import reward
    except Exception as exc_:
        print(INFO_CUSTOM_REWARD)
        reward = RedispReward
    if not isinstance(reward, type):
        raise RuntimeError(REWARD_ERR)
    if not issubclass(reward, BaseReward):
        raise RuntimeError(REWARD_ERR2)
    try:
        from submission import other_rewards
    except Exception as exc_:
        print(INFO_CUSTOM_OTHER)
        other_rewards = {}
        
    # add the other rewards to compute the real score
    key_score = config["score_config"]["key_score"]
    if key_score in other_rewards:
        print(KEY_OVERLOAD_WARN.format(key_score))
    other_rewards[key_score] = L2RPNWCCI2022ScoreFun

    # create the real environment
    real_env = grid2op.make(input_dir,
                            reward_class=reward,
                            other_rewards=other_rewards,
                            backend=backend_cls(),
                            )

    # this is called after, so that no one can change this sequence
    np.random.seed(int(config["score_config"]["seed"]))
    max_int = np.iinfo(dt_int).max
    # env seeds are read from the json
    env_seeds =  [int(config["episodes_info"][os.path.split(el)[-1]]["seed"]) for el in sorted(real_env.chronics_handler.real_data.subpaths)]
    # agent seeds are generated with the provided random seed
    agent_seeds = list(np.random.randint(max_int, size=int(config["nb_scenario"])))
    path_save = os.path.abspath(output_dir)
    scores = ScoreL2RPN2022(env=real_env,
                            env_seeds=env_seeds,
                            agent_seeds=agent_seeds,
                            nb_scenario=int(config["nb_scenario"]),
                            min_losses_ratio=float(config["score_config"]["min_losses_ratio"]),
                            verbose=0 if not args.verbose else 2,
                            max_step=-1,
                            nb_process_stats=1)
    print(STARTING_THE_EVALUATION)
    beg_ = time.perf_counter()
    scores, n_played, total_ts = scores.get(submitted_agent, path_save=path_save, nb_process=1)
    res_scores = {"scores": [float(score) for score in scores],
                  "n_played": [int(el) for el in n_played],
                  "total_ts": [int(el) for el in total_ts]}
    end_ = time.perf_counter()
    print(f"[INFO] agent scoring time time: {end_ - beg_:.2f}s")
    print(ENDING_THE_EVALUATION)
    with open(os.path.join(path_save, "res_agent.json"), "w", encoding="utf-8") as f:
        json.dump(obj=res_scores, fp=f)

    if args.gif_episode is not None:
        beg_ = time.perf_counter()
        gif_input = os.path.join(output_dir)
        write_gif(output_dir, gif_input, args.gif_episode,
                  args.gif_start, args.gif_end)
        end_ = time.perf_counter()
        print(f"[INFO] gif writing time: {end_ - beg_:.2f}s")
    env_template.close()
    real_env.close()

    if args.cleanup:
        cmds = [
            "find {} -name '*.npz' | xargs -i rm -rf {}",
            "find {} -name 'dict_*.json' | xargs -i rm -rf {}",
            "find {} -name '_parameters.json' | xargs -i rm -rf {}"
        ]
        for cmd in cmds:
            os.system(cmd.format(output_dir, "{}"))
    print("Done and data saved in : \"{}\"".format(path_save))


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        print("ERROR: ingestion program failed with error: \n{}\n".format(e))
        print("------------------------------------")
        print("        Detailed error Logs         ")
        print("------------------------------------")
        traceback.print_exc(file=sys.stdout)
        print("------------------------------------")
        print("      End Detailed error Logs       ")
        print("------------------------------------")
        sys.exit(1)
