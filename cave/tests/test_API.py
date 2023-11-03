from ..TrainTestAPI import maker_TrainTestAPI
from ..Environment import Environment
from ..import KEYWORD
import shutil

def test_TrainTestAPI_Train():
    model_filename = "model"
    env_id = "Pendulum-v1"
    algo = "PPO"
    tmp_dirpath = "pytest_tmp"

    algo_kwargs = dict(n_steps=8, batch_size=2,
        policy="MlpPolicy",
        policy_kwargs=dict(net_arch=dict(pi=[32, 16], vf=[32, 16])))

    reward_api = f'''
def {Environment.IS_VIOLATED_FUNC_ID}(x, y):
    return False, False

def {Environment. GET_REWARD_FUNC_ID}(x, y, reward, violated):
    return reward
    '''
    env_kwargs = dict()

    api = maker_TrainTestAPI(env_id=env_id,
                 env_kwargs=env_kwargs,
                 algo=algo,
                 algo_kwargs=algo_kwargs,
                 model_filename=model_filename,
                 #   curr_model_dirpath="tmp1",
                 next_model_dirpath=tmp_dirpath,
                 onnx_filename=model_filename,
                 reward_api=reward_api,
                 test_log_filename="test.log",
                 total_cycle=100,
                 mode=KEYWORD.TRAIN,
                 nproc=2)

    api()
    shutil.rmtree(tmp_dirpath, ignore_errors=True)
