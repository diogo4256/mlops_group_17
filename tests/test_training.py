import omegaconf
from tests import _PROJECT_ROOT
import os
import torch
import src.train_model as train
from unittest.mock import patch, MagicMock
from torch.utils.data import DataLoader

def test_custom_dataset():
    # Simulates a dataset
    images = torch.rand(10, 3, 100, 100)
    labels = torch.randint(0, 1, (10,))
    dataset = train.CustomDataset(images, labels)
    assert len(dataset) == 10
    image, label = dataset[0]
    assert torch.equal(image, images[0])
    assert label == labels[0]
    torch.save(images, "tests/data/processed/mock/fruit_training_images.pt")
    torch.save(labels, "tests/data/processed/mock/fruit_training_labels.pt")

def test_load_data():
    # Arrange
    hparams = {
        'processed_dataset': 'tests/data/processed',
        'dataset_name': 'mock',
        'batch_size': 32,
        'shuffle': True
    }

    try:
        # Act
        result = train.load_data(_PROJECT_ROOT, hparams)

        # Assert
        assert isinstance(result, DataLoader)
        assert result.batch_size == hparams['batch_size']
        
    finally:
        # Cleanup
        os.remove("tests/data/processed/mock/fruit_training_images.pt")
        os.remove("tests/data/processed/mock/fruit_training_labels.pt")

def test_setup_model_and_optimizer():
    # Arrange
    hparams = {
        'learning_rate': 0.01,
        'weight_decay': 0.0005
    }

    # Act
    model, criterion, optimizer = train.setup_model_and_optimizer(hparams)

    # Assert
    assert isinstance(model, torch.nn.Module)
    assert model.__class__.__name__ == 'ResNet'
    assert isinstance(criterion, torch.nn.CrossEntropyLoss)
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.defaults['lr'] == 0.01
    assert optimizer.defaults['weight_decay'] == 0.0005
    
@patch('wandb.log')
@patch('torch.utils.data.DataLoader', autospec=True)
@patch('torch.nn.CrossEntropyLoss', autospec=True)
@patch('torch.optim.SGD', autospec=True)
@patch('torch.nn.Module', autospec=True)
def test_train_one_epoch(mock_module, mock_optimizer, mock_criterion, mock_trainloader, mock_wandb_log):
    # Arrange
    model = mock_module
    criterion = mock_criterion
    optimizer = mock_optimizer
    trainloader = mock_trainloader
    trainloader.__len__.return_value = 10

    # Act
    result = train.train_one_epoch(model, criterion, optimizer, trainloader)

    # Assert
    assert isinstance(result, float) or isinstance(result, int)
    
@patch('wandb.log')
@patch('src.train_model.plot_training_loss', autospec=True)
@patch('src.train_model.train_one_epoch', autospec=True)
@patch('src.train_model.setup_model_and_optimizer', autospec=True)
@patch('src.train_model.load_data', autospec=True)
@patch('hydra.core.hydra_config.HydraConfig.get', autospec=True)
@patch('torch.save', autospec=True)
def test_train(mock_torch_save, mock_get, mock_load_data, 
               mock_setup_model_and_optimizer, mock_train_one_epoch, 
               mock_plot_training_loss, mock_wandb_log):
    # Arrange
    mock_get.return_value.runtime.cwd = '.'
    mock_load_data.return_value = [torch.rand(10, 3, 224, 224), torch.rand(10)]
    mock_model = MagicMock()
    mock_model.state_dict.return_value = MagicMock()
    mock_setup_model_and_optimizer.return_value = (mock_model, MagicMock(), MagicMock())
    mock_train_one_epoch.return_value = 0.5
    config = omegaconf.OmegaConf.create()
    
    config.train = {
        'epochs': 10,
        'model_savepath': 'models/trained_model.pth'
    }
    
    config.wandb = {
        'project': 'fruit-tests',
        'entity': 'diogo-adegas'
    }

    # Act
    train.train(config)

    # Assert
    mock_torch_save.assert_called_once_with(mock_model.state_dict.return_value, os.path.join(mock_get.return_value.runtime.cwd, config.train["model_savepath"]))