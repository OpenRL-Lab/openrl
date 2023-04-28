from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseVecInfo(ABC):
    def __init__(self, parallel_env_num: int, agent_num: int):
        super(BaseVecInfo, self).__init__()
        self.parallel_env_num = parallel_env_num
        self.agent_num = agent_num

    @abstractmethod
    def statistics(self, buffer: Any) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def append(self, info: Dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError
