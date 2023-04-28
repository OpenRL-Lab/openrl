from typing import Any

from openrl.envs.vec_env.base_venv import BaseVecEnv
from openrl.envs.vec_env.vec_info.simple_vec_info import SimpleVecInfo

registed_vec_info = {
    "default": SimpleVecInfo,
}


class VecInfoFactory:
    @staticmethod
    def get_vec_info_class(vec_info_class: Any, env: BaseVecEnv):
        VecInfoFactory.auto_register(vec_info_class)
        if vec_info_class is None or vec_info_class.id is None:
            return registed_vec_info["default"](env.parallel_env_num, env.agent_num)
        return registed_vec_info[vec_info_class.id](
            env.parallel_env_num, env.agent_num, **vec_info_class.args
        )

    @staticmethod
    def register(name: str, vec_info: Any):
        registed_vec_info[name] = vec_info

    @staticmethod
    def auto_register(vec_info_class: Any):
        if vec_info_class is None:
            return
        elif vec_info_class.id == "NLPVecInfo":
            from openrl.envs.vec_env.vec_info.nlp_vec_info import NLPVecInfo

            VecInfoFactory.register("NLPVecInfo", NLPVecInfo)
