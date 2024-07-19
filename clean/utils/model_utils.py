
from .DAVE2pytorch import *
def load_model(name, logger):
    if 'v1' in name:
        logger.info('Loading DAVE2v1 model')
        model = DAVE2v1()
        weights = model.load(path='./models/log/DAVE2_v1_center/Aug=True/weights_10000_best.pth')
        model.load_state_dict(weights)
    elif 'v2' in name:
        logger.info('Loading DAVE2v2 model')
        model = DAVE2v2()
        weights = model.load(path='./models/log/DAVE2_v2_center/Aug=True/weights_5000_best.pth')
        model.load_state_dict(weights)
    elif 'v3' in name:
        logger.info('Loading DAVE2v3 model')
        model = DAVE2v3()
        weights = model.load(path="./models/log/DAVE2_v3_center/Aug=True/weights_5000_best.pth")
        model.load_state_dict(weights)
    elif 'epoch' in name:
        logger.info('Loading Epoch model')
        model = Epoch()
        weights = model.load(path="./models/log/epoch/Aug=True/weights_10000_best.pth")
        model.load_state_dict(weights)
    else:
        model = None
    return model