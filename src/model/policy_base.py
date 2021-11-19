import abc


class BasePolicy(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def control(self, observation):
        raise NotImplementedError
