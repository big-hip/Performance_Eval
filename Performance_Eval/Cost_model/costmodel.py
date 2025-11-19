import random


class CostModel:
    """
        args:
            type: type of the operator
            list: list of this operator's input
        
        return:
            int: performance evaluation of the operator
            
    """
    call_counter = 1

    def __init__(self, type: str, input:list =[]):
        self.type = type
        self.input = input
        pass

    def __call__(self, *args, **kwds) -> None:
        self.call_counter += 1


