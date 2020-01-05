
# My Libraries
from model.net import Net
from src.dataset import FaceDataset
from src.loadConfig import loadConfig

if __name__ == "__main__":
    train_dir = r"F:\DataSets\toy"
    model_dir = './checkpoints/model.dat'
    dst_dir = './checkpoints/model_transfered.dat'
    fdata = FaceDataset(dir=train_dir)
    config = loadConfig('config.json')

    NN = Net(config=config)
    NN.load(model_dir)
    NN.train(fdata)
    NN.save(dst_dir)
