from deeprl import models  # noqa
from deeprl.build_graph import build_act, build_train_att, build_train_def  # noqa
# from deeprl.deepq import learn, load_act  # noqa
from deeprl.replay_buffer import ReplayBuffer  # noqa

def wrap_atari_dqn(env):
    from common.atari_wrappers import wrap_deepmind
    return wrap_deepmind(env, frame_stack=True, scale=False)
