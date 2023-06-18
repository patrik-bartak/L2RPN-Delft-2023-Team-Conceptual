import grid2op
from submission_withbaselines.my_baseline import make_agent

if __name__ == "__main__":
  # env = grid2op.make("l2rpn_wcci_2022_dev")
  env = grid2op.make("l2rpn_wcci_2022")
  agent = make_agent(env, "C:/Users/patri/code/patrik-bartak/L2RPN-Delft-2023-Team-Conceptual/example_submissions/submission_withbaselines")
  print(agent.act(env.reset(), 0, False))
