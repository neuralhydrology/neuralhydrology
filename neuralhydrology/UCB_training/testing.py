import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
from neuralhydrology.UCB_training import UCB_train
from UCB_train import UCB_trainer  # Ensure this import path is correct

class TestUCBTrainer(unittest.TestCase):
    def setUp(self):
        """Setup method to create a UCB_trainer object for testing."""
        self.csv_path = '/Users/evanrobert/Documents/ESDL_Research/russian_river/UCB-USACE-LSTMs/neuralhydrology/UCB_training/tuler_training_data.csv'
        self.hyperparameters = {'learning_rate': 0.001, 'batch_size': 256, 'epochs': 16}
        self.num_ensemble_members = 8
        self.trainer = UCB_trainer(self.csv_path, self.hyperparameters, self.num_ensemble_members)

    @patch('UCB_trainer.Config')  # Patch the Config class from the correct path
    def test_create_config(self, mock_config):
        """Test if config object is correctly created and updated."""
        mock_config_obj = MagicMock()
        mock_config.return_value = mock_config_obj
        self.trainer._create_config()
        mock_config_obj.update_config.assert_called_once_with(self.trainer.hyperparams)
        self.assertIsNotNone(self.trainer.config)

    @patch('UCB_trainer.Path')  # Patch Path
    def test_train_model(self, mock_path):
        """Test if train_model returns a valid path."""
        mock_path.return_value = Path('/fake/path/model_result')
        result_path = self.trainer._train_model()
        self.assertIsInstance(result_path, Path)
        self.assertEqual(result_path, mock_path.return_value)

    def test_get_metrics(self):
        """Test if the metrics are returned as a dictionary."""
        # Assuming _get_metrics is returning a dict
        metrics = self.trainer._get_metrics()
        self.assertIsInstance(metrics, dict)

    def test_generate_plots(self):
        """Test if the plotting methods are called correctly."""
        # Patch plotting methods to check if they are called
        with patch.object(self.trainer, '_generate_plot1') as mock_plot1, \
             patch.object(self.trainer, '_generate_plot2') as mock_plot2:
            self.trainer._generate_plot1()
            self.trainer._generate_plot2()
            mock_plot1.assert_called_once()
            mock_plot2.assert_called_once()

    def test_results(self):
        """Test if results method returns the correct output."""
        with patch.object(self.trainer, '_get_metrics') as mock_get_metrics, \
             patch.object(self.trainer, '_generate_plot1') as mock_plot1, \
             patch.object(self.trainer, '_generate_plot2') as mock_plot2:
            mock_get_metrics.return_value = {'accuracy': 0.9}  # Example mock return
            result = self.trainer.results()

            # Ensure metrics and plots are generated
            mock_get_metrics.assert_called_once()
            mock_plot1.assert_called_once()
            mock_plot2.assert_called_once()

            # Check the result type and content
            self.assertIsInstance(result, dict)
            self.assertIn('accuracy', result)


if __name__ == '__main__':
    unittest.main()
