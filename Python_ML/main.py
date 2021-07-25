from dataset_creators import create_dataset
from learning import training, inference
from model import get_model, get_spectrogram_model
from save_model import save_model


if __name__ == "__main__":
    # Create the datasets which have 20% validation set
    train_dl, val_dl = create_dataset(0.2)
    # Train Epochs
    num_epochs = 50
    # Create the untrained model
    myModel, device = get_model()
    # Train the model on the training set
    myModel = training(myModel, train_dl, num_epochs, device)
    # report on the validation set
    inference(myModel, val_dl, device)
    # save the model to be deployed
    save_model(myModel)
    # If the error: RuntimeError: fft: ATen not compiled with MKL support is fixed, the next can be applied
    # The specModel is build to be used as a preprocessing step on the recorded .wav files

    #specModel = get_spectrogram_model()
    #save_model(specModel, type='Spec')
