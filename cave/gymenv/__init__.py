import gymnasium as gym

gym.register(id='NetworkCC-v0', entry_point='cave.gymenv.networkcc_v0.Env:Env')
gym.register(id='NetworkCC-v1', entry_point='cave.gymenv.networkcc_v1.Env:Env')
gym.register(id='B1-v0', entry_point='cave.gymenv.b1_v0.Env:Env')
gym.register(id='B2-v0', entry_point='cave.gymenv.b2_v0.Env:Env')
gym.register(id='Tora-v0', entry_point='cave.gymenv.tora_v0.Env:Env')