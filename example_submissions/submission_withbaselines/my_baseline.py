from l2rpn_baselines.PPO_SB3 import evaluate
import json
import os
from .CustomGymEnv import CustomGymEnv
from grid2op.Agent import BaseAgent

name = "PPO_agent"

class BaselineAgent(BaseAgent):
  def __init__(self, l2rpn_agent):
    self.l2rpn_agent = l2rpn_agent
    BaseAgent.__init__(self, l2rpn_agent.action_space)
  
  def act(self, obs, reward, done=False):
    action = self.l2rpn_agent.act(obs, reward, done)
    # We try to limit to end up with a "game over" because actions on curtailment or storage units.
    action.limit_curtail_storage(obs, margin=150)
    return action


def make_agent(env, submission_dir):
    """
    This function will be used by codalab to create your agent. It should accept exactly an environment and a path
    to your submission directory and return a valid agent.
    """

    agent_dir = os.path.join(submission_dir, name)

    with open(os.path.join(agent_dir, "preprocess_obs.json"), 'r', encoding="utf-8") as f:
      obs_space_kwargs = json.load(f)
    with open(os.path.join(agent_dir, "preprocess_act.json"), 'r', encoding="utf-8") as f:
      act_space_kwargs = json.load(f)

    l2rpn_agent, _ = evaluate(env,
                    nb_episode=0,
                    load_path=submission_dir,
                    name=name,
                    gymenv_class=CustomGymEnv,
                    gymenv_kwargs={"safe_max_rho": 0.9},
                    obs_space_kwargs=obs_space_kwargs,
                    act_space_kwargs=act_space_kwargs)

    return BaselineAgent(l2rpn_agent)