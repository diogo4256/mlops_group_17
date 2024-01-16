from tests import _PATH_DATA
import os
import unittest
from unittest.mock import patch, MagicMock
import torch
from torchvision import transforms
from src.train_model import train

path_to_data = os.path.join(_PATH_DATA, "raw/fruits-360_dataset/fruits-360/Training")
subfolder = "processed_data/fruits_dataset"
path_extract = os.path.join(_PATH_DATA, "raw")
image_paths = path_extract + "/fruits-360_dataset/fruits-360/Training/Apple Braeburn"

class TestTrainModel(unittest.TestCase):
    
    @patch('train_model.timm.create_model')
    @patch('train_model.torch.load')
    @patch('train_model.optim.SGD')
    @patch('train_model.torch.save')
    @patch('train_model.os.makedirs')
    @patch('train_model.os.path.join')
    @patch('train_model.os.getcwd')
    @patch('train_model.plt.savefig')
    @patch('train_model.plt.plot')
    def test_train(self, mock_plot, mock_savefig, mock_getcwd, mock_osjoin,
                   mock_makedirs, mock_torchsave, mock_optim, mock_torchload, mock_create_model):
        mock_create_model.return_value = MagicMock()
        mock_torchload.side_effect = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
        mock_getcwd.return_value = '/Users/panagiotaemmanouilidi/Desktop/MS1/Jan2024/mlops_group_17/tests'
        
        # Mocking HydraConfig.get().runtime.cwd
        with patch('train_model.hydra.core.hydra_config.HydraConfig.get') as mock_hydra_get:
            mock_hydra_get.return_value.runtime.cwd = '/Users/panagiotaemmanouilidi/Desktop/MS1/Jan2024/mlops_group_17/tests/test_training.py'

            # Mocking hyperparameters
            config = {'train': {'epochs': 2, 'batch_size': 64, 'shuffle': True,
                                'learning_rate': 0.001, 'weight_decay': 0.0001,
                                'dropout_rate': 0.5, 'processed': 'data',
                                'fruits-360_dataset': 'Apple Braeburn'}}
            
            with patch('train_model.hydra.main', return_value=config):
                train(config)
        
        # Assertions
        mock_create_model.assert_called_once_with('resnet50', pretrained=False)
        mock_torchload.assert_called_with('../data/raw/fruits-360_dataset/fruits-360/Training/Apple Braeburn')     #'../fruits-360_dataset/fruits-360/Training/Apple Braeburn'
        mock_optim.assert_called_once_with(MagicMock().parameters(), lr=0.001, weight_decay=0.0001)
        mock_plot.assert_called_once()
        mock_savefig.assert_called_once_with('../reports/figures/training_loss.png')
        mock_torchsave.assert_called_once_with(MagicMock().state_dict(), '../models/trained_model.pth')
        mock_makedirs.assert_called_once_with('../models', exist_ok=True)


if __name__ == '__main__':
    unittest.main()