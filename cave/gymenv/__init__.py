import gymnasium as gym

gym.register(id='NetworkCC-v0', entry_point='cave.gymenv.networkcc_v0.Env:Env')
gym.register(id='NetworkCC-v1', entry_point='cave.gymenv.networkcc_v1.Env:Env')