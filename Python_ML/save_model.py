import torch
from torch.utils.mobile_optimizer import optimize_for_mobile


def save_model(model, type_str='Classifier'):
    # Save the torchscript model to be used for predictions later
    print('Saving model...')
    torchscript_model = torch.jit.script(model)
    optimized_torchscript_model = optimize_for_mobile(torchscript_model)
    if type_str == 'Classifier':
        optimized_torchscript_model.save("optimized_torchscript_model.pt")
    else:
        optimized_torchscript_model.save("spectrogram_model.pt")
    return

