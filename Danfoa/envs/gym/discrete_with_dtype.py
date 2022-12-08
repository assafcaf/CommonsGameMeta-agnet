from gym.spaces import Discrete


class DiscreteWithDType(Discrete):
    def __init__(self, n, dtype):
        # Skip Discrete __init__ on purpose, to avoid setting the wrong dtype
        self.dtype = dtype
        super().__init__(n)
