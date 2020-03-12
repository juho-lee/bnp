from models.np import CNP, NP, BNP
from models.anp import CANP, ANP, BANP
from models.testbed import ANP2, BANP2

def load_model(model_name, **kwargs):
    if model_name == 'cnp':
        return CNP(**kwargs)
    elif model_name == 'np':
        return NP(**kwargs)
    elif model_name == 'bnp':
        return BNP(**kwargs)
    elif model_name == 'canp':
        return CANP(**kwargs)
    elif model_name == 'anp':
        return ANP(**kwargs)
    elif model_name == 'anp2':
        return ANP2(**kwargs)
    elif model_name == 'banp':
        return BANP(**kwargs)
    elif model_name == 'banp2':
        return BANP2(**kwargs)
