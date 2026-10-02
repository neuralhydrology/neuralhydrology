import logging
import pickle
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import neuralhydrology.training.loss as loss
from neuralhydrology.datasetzoo import get_dataset
from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.datautils.utils import load_basin_file, load_scaler
from neuralhydrology.evaluation import get_tester
from neuralhydrology.evaluation.tester import BaseTester
from neuralhydrology.modelzoo import get_model
from neuralhydrology.training import get_loss_obj, get_optimizer, get_regularization_obj
from neuralhydrology.training.logger import Logger
from neuralhydrology.utils.config import Config
from neuralhydrology.utils.logging_utils import setup_logging
from neuralhydrology.training.earlystopper import EarlyStopper

LOGGER = logging.getLogger(__name__)


class BaseTrainer(object):
    """Default class to train a model.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """

    def __init__(self, cfg: Config):
        super(BaseTrainer, self).__init__()
        self.cfg = cfg
        self.model = None
        self.optimizer = None
        self.loss_obj = None
        self.experiment_logger = None
        self.loader = None
        self.validator = None
        self.noise_sampler_y = None
        self._target_mean = None
        self._target_std = None
        self._scaler = {}
        self._allow_subsequent_nan_losses = cfg.allow_subsequent_nan_losses
        self._disable_pbar = cfg.verbose == 0
        self._max_updates_per_epoch = cfg.max_updates_per_epoch
        self._early_stopping = cfg.early_stopping
        self._patience_early_stopping = cfg.patience_early_stopping
        self._minimum_epochs_before_early_stopping = cfg.minimum_epochs_before_early_stopping
        self._dynamic_learning_rate = cfg.dynamic_learning_rate
        self._patience_dynamic_learning_rate = cfg.patience_dynamic_learning_rate
        self._factor_dynamic_learning_rate = cfg.factor_dynamic_learning_rate

        # load train basin list and add number of basins to the config
        self.basins = load_basin_file(cfg.train_basin_file)
        self.cfg.number_of_basins = len(self.basins)

        # check at which epoch the training starts
        self._epoch = self._get_start_epoch_number()

        self._create_folder_structure()
        setup_logging(str(self.cfg.run_dir / "output.log"))
        LOGGER.info(f"### Folder structure created at {self.cfg.run_dir}")

        if self.cfg.is_continue_training:
            LOGGER.info(f"### Continue training of run stored in {self.cfg.base_run_dir}")

        if self.cfg.is_finetuning:
            LOGGER.info(f"### Start finetuning with pretrained model stored in {self.cfg.base_run_dir}")

        LOGGER.info(f"### Run configurations for {self.cfg.experiment_name}")
        for key, val in self.cfg.as_dict().items():
            LOGGER.info(f"{key}: {val}")

        self._set_random_seeds()
        self._set_device()

    def _get_dataset(self) -> BaseDataset:
        return get_dataset(cfg=self.cfg, period="train", is_train=True, scaler=self._scaler)

    def _get_model(self) -> torch.nn.Module:
        return get_model(cfg=self.cfg)

    def _get_optimizer(self) -> torch.optim.Optimizer:
        return get_optimizer(model=self.model, cfg=self.cfg)

    def _get_loss_obj(self) -> loss.BaseLoss:
        return get_loss_obj(cfg=self.cfg)

    def _set_regularization(self):
        self.loss_obj.set_regularization_terms(get_regularization_obj(cfg=self.cfg))

    def _raw_model(self) -> torch.nn.Module:
        """Return the model without its torch.compile() wrapper, if any.

        torch.compile() wraps the model in an OptimizedModule whose state_dict keys are
        prefixed with '_orig_mod.'. Checkpoints must always be saved/loaded through the
        wrapped module directly, so that files on disk have the same keys regardless of
        whether the model happens to be compiled — otherwise continue_training, finetuning,
        and evaluation (which never compiles) can't read each other's checkpoints.
        """
        return getattr(self.model, '_orig_mod', self.model)

    def _get_tester(self) -> BaseTester:
        return get_tester(cfg=self.cfg, run_dir=self.cfg.run_dir, period="validation", init_model=False)

    def _get_data_loader(self, ds: BaseDataset) -> torch.utils.data.DataLoader:
        return DataLoader(ds,
                          batch_size=self.cfg.batch_size,
                          shuffle=True,
                          num_workers=self.cfg.num_workers,
                          collate_fn=ds.collate_fn,
                          # pinned memory lets CUDA transfer batches asynchronously while the GPU is busy
                          pin_memory=self.device.type == 'cuda',
                          # keeps worker processes alive between epochs instead of restarting them every time
                          persistent_workers=self.cfg.num_workers > 0)

    def _freeze_model_parts(self):
        # freeze all model weights
        for param in self.model.parameters():
            param.requires_grad = False

        unresolved_modules = []

        # unfreeze parameters specified in config as tuneable parameters
        if isinstance(self.cfg.finetune_modules, list):
            for module_part in self.cfg.finetune_modules:
                if module_part in self.model.module_parts:
                    module = getattr(self.model, module_part)
                    for param in module.parameters():
                        param.requires_grad = True
                else:
                    unresolved_modules.append(module_part)
        else:
            # if it was no list, it has to be a dictionary
            for module_group, module_parts in self.cfg.finetune_modules.items():
                if module_group in self.model.module_parts:
                    if isinstance(module_parts, str):
                        module_parts = [module_parts]
                    for module_part in module_parts:
                        module = getattr(self.model, module_group)[module_part]
                        for param in module.parameters():
                            param.requires_grad = True
                else:
                    unresolved_modules.append(module_group)
        if unresolved_modules:
            LOGGER.warning(f"Could not resolve the following module parts for finetuning: {unresolved_modules}")

    def initialize_training(self):
        """Initialize the training class.

        This method will load the model, initialize loss, regularization, optimizer, dataset and dataloader,
        tensorboard logging, and Tester class.
        If called in a ``continue_training`` context, this model will also restore the model and optimizer state.
        """
        if self.cfg.is_finetuning:
            # Load scaler from pre-trained model.
            self._scaler = load_scaler(self.cfg.base_run_dir)

        # Initialize dataset before the model is loaded.
        ds = self._get_dataset()
        if len(ds) == 0:
            raise ValueError("Dataset contains no samples.")
        self.loader = self._get_data_loader(ds=ds)

        self.model = self._get_model().to(self.device)

        # torch.compile() fuses and optimizes the computation graph — first epoch is slower (compilation),
        # all subsequent ones are faster. Only available on PyTorch >= 2.0 and worthwhile only on CUDA.
        # Opt-out via `compile_model: False` in the run config: compilation needs a writable disk
        # cache for Triton's generated kernels, which can fail on clusters with tight disk quotas.
        if self.cfg.compile_model and hasattr(torch, 'compile') and self.device.type == 'cuda':
            self.model = torch.compile(self.model)
            LOGGER.info("Model compiled with torch.compile()")

        if self.cfg.checkpoint_path is not None:
            LOGGER.info(f"Starting training from Checkpoint {self.cfg.checkpoint_path}")
            # weights_only=False needed for PyTorch >= 2.6 compatibility with optimizer states
            self._raw_model().load_state_dict(
                torch.load(str(self.cfg.checkpoint_path), map_location=self.device, weights_only=False))
        elif self.cfg.checkpoint_path is None and self.cfg.is_finetuning:
            # the default for finetuning is the last model state
            checkpoint_path = [x for x in sorted(list(self.cfg.base_run_dir.glob('model_epoch*.pt')))][-1]
            LOGGER.info(f"Starting training from checkpoint {checkpoint_path}")
            self._raw_model().load_state_dict(
                torch.load(str(checkpoint_path), map_location=self.device, weights_only=False))

        # Freeze model parts from pre-trained model.
        if self.cfg.is_finetuning:
            self._freeze_model_parts()

        self.optimizer = self._get_optimizer()
        self.loss_obj = self._get_loss_obj().to(self.device)

        # Add possible regularization terms to the loss function.
        self._set_regularization()

        # restore optimizer and model state if training is continued
        if self.cfg.is_continue_training:
            self._restore_training_state()

        self.experiment_logger = Logger(cfg=self.cfg)
        if self.cfg.log_tensorboard:
            self.experiment_logger.start_tb()

        if self.cfg.is_continue_training:
            # set epoch and iteration step counter to continue from the selected checkpoint
            self.experiment_logger.epoch = self._epoch
            self.experiment_logger.update = len(self.loader) * self._epoch

        if self.cfg.validate_every is not None:
            if self.cfg.validate_n_random_basins < 1:
                warn_msg = [
                    f"Validation set to validate every {self.cfg.validate_every} epoch(s), but ",
                    "'validate_n_random_basins' not set or set to zero. Will validate on the entire validation set."
                ]
                LOGGER.warning("".join(warn_msg))
                self.cfg.validate_n_random_basins = self.cfg.number_of_basins
            self.validator = self._get_tester()

        if self.cfg.target_noise_std is not None:
            self.noise_sampler_y = torch.distributions.Normal(loc=0, scale=self.cfg.target_noise_std)
            self._target_mean = torch.from_numpy(
                ds.scaler["xarray_feature_center"][self.cfg.target_variables].to_array().values).to(self.device)
            self._target_std = torch.from_numpy(
                ds.scaler["xarray_feature_scale"][self.cfg.target_variables].to_array().values).to(self.device)

    def train_and_validate(self):
        """Train and validate the model.

        Train the model for the number of epochs specified in the run configuration, and perform validation after every
        ``validate_every`` epochs. Model and optimizer state are saved after every ``save_weights_every`` epochs.
        """
        if self._early_stopping:
            if self.cfg.is_continue_training:
                LOGGER.warning("Early stopping state is reset.")
            early_stopper = EarlyStopper(patience = self._patience_early_stopping, min_delta = 0.0001)

        if self._dynamic_learning_rate:
            if self.cfg.is_continue_training:
                LOGGER.warning("Scheduler state is reset.")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=self._factor_dynamic_learning_rate, patience=self._patience_dynamic_learning_rate)

        # cfg.epochs is the total target epoch, not the number of additional epochs to run.
        # This way continue_training works correctly without modifying the config between sessions.
        if self._epoch >= self.cfg.epochs:
            LOGGER.info(f"Already at epoch {self._epoch}, target is {self.cfg.epochs}. Nothing to train.")
            return

        for epoch in range(self._epoch + 1, self.cfg.epochs + 1):
            if not self._dynamic_learning_rate:
                if epoch in self.cfg.learning_rate.keys():
                    LOGGER.info(f"Setting learning rate to {self.cfg.learning_rate[epoch]}")
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.cfg.learning_rate[epoch]

            self._train_epoch(epoch=epoch)
            avg_losses = self.experiment_logger.summarise()
            loss_str = ", ".join(f"{k}: {v:.5f}" for k, v in avg_losses.items())
            LOGGER.info(f"Epoch {epoch} average loss: {loss_str}")

            if epoch % self.cfg.save_weights_every == 0:
                self._save_weights_and_optimizer(epoch)

            if (self.validator is not None) and (epoch % self.cfg.validate_every == 0):
                self.validator.evaluate(epoch=epoch,
                                        save_results=self.cfg.save_validation_results,
                                        save_all_output=self.cfg.save_all_output,
                                        metrics=self.cfg.metrics,
                                        model=self.model,
                                        experiment_logger=self.experiment_logger.valid())

                valid_metrics = self.experiment_logger.summarise()
                print_msg = f"Epoch {epoch} average validation loss: {valid_metrics['avg_total_loss']:.5f}"
                if self.cfg.metrics:
                    print_msg += f" -- Median validation metrics: "
                    print_msg += ", ".join(f"{k}: {v:.5f}" for k, v in valid_metrics.items() if k != 'avg_total_loss')
                    LOGGER.info(print_msg)
                

                if self._early_stopping and epoch > self._minimum_epochs_before_early_stopping and early_stopper.check_early_stopping(valid_metrics['avg_total_loss']):
                    LOGGER.info(f"Early stopping triggered at epoch {epoch} with validation loss {valid_metrics['avg_total_loss']:.5f}. Training stopped.")
                    break
                if self._dynamic_learning_rate:
                    scheduler.step(valid_metrics['avg_total_loss'])

        # make sure to close tensorboard to avoid losing the last epoch
        if self.cfg.log_tensorboard:
            self.experiment_logger.stop_tb()

    def _find_latest_checkpoint(self, search_dir: Path) -> Path:
        """Find the most recent model checkpoint in a run directory.

        Searches both the run directory itself and any continue_training subfolders,
        since continued runs save their weights one level deeper.
        """
        all_weights = list(search_dir.glob('model_epoch*.pt'))
        # also pick up weights from previous continue_training sessions
        all_weights += list(search_dir.glob('continue_training_from_epoch*/model_epoch*.pt'))
        if not all_weights:
            raise FileNotFoundError(f"No model checkpoint found in {search_dir}")
        # sort by epoch number in the filename, not by path (paths differ between root and subfolders)
        return max(all_weights, key=lambda p: int(p.stem[-3:]))

    def _get_start_epoch_number(self):
        if self.cfg.is_continue_training:
            if self.cfg.continue_from_epoch is not None:
                epoch = self.cfg.continue_from_epoch
            else:
                weight_path = self._find_latest_checkpoint(self.cfg.run_dir)
                epoch = weight_path.stem[-3:]
        else:
            epoch = 0
        return int(epoch)

    def _restore_training_state(self):
        if self.cfg.continue_from_epoch is not None:
            epoch = f"{self.cfg.continue_from_epoch:03d}"
            weight_path = self.cfg.base_run_dir / f"model_epoch{epoch}.pt"
        else:
            weight_path = self._find_latest_checkpoint(self.cfg.base_run_dir)
            epoch = weight_path.stem[-3:]

        optimizer_path = weight_path.parent / f"optimizer_state_epoch{epoch}.pt"

        LOGGER.info(f"Continue training from epoch {int(epoch)}")
        self._raw_model().load_state_dict(torch.load(weight_path, map_location=self.device, weights_only=False))
        self.optimizer.load_state_dict(torch.load(str(optimizer_path), map_location=self.device, weights_only=False))

    def _save_weights_and_optimizer(self, epoch: int):
        weight_path = self.cfg.run_dir / f"model_epoch{epoch:03d}.pt"
        torch.save(self._raw_model().state_dict(), str(weight_path))

        optimizer_path = self.cfg.run_dir / f"optimizer_state_epoch{epoch:03d}.pt"
        torch.save(self.optimizer.state_dict(), str(optimizer_path))

    def _train_epoch(self, epoch: int):
        self.model.train()
        self.experiment_logger.train()

        # process bar handle
        n_iter = min(self._max_updates_per_epoch, len(self.loader)) if self._max_updates_per_epoch is not None else None
        pbar = tqdm(self.loader, file=sys.stdout, disable=self._disable_pbar, total=n_iter)
        pbar.set_description(f'# Epoch {epoch}')

        # Iterate in batches over training set
        nan_count = 0
        for i, data in enumerate(pbar):
            if self._max_updates_per_epoch is not None and i >= self._max_updates_per_epoch:
                break

            # non_blocking=True lets the transfer overlap with GPU computation on the previous batch
            for key in data.keys():
                if key.startswith('x_d'):
                    data[key] = {k: v.to(self.device, non_blocking=True) for k, v in data[key].items()}
                elif not key.startswith('date'):
                    data[key] = data[key].to(self.device, non_blocking=True)

            # apply possible pre-processing to the batch before the forward pass
            data = self.model.pre_model_hook(data, is_train=True)

            # autocast runs the forward pass and loss in reduced precision (BF16 or FP16 on CUDA),
            # which uses the GPU's tensor cores and roughly doubles throughput on modern hardware
            with torch.autocast(device_type=self.device.type, dtype=self._amp_dtype, enabled=self._use_amp):
                predictions = self.model(data)

                if self.noise_sampler_y is not None:
                    for key in filter(lambda k: 'y' in k, data.keys()):
                        noise = self.noise_sampler_y.sample(data[key].shape)
                        # make sure we add near-zero noise to originally near-zero targets
                        data[key] += (data[key] + self._target_mean / self._target_std) * noise.to(self.device)

                loss, all_losses = self.loss_obj(predictions, data)

            # early stop training if loss or gradients are NaN/Inf
            if torch.isnan(loss):
                step_ok = False
            else:
                self.optimizer.zero_grad()

                if self._grad_scaler is not None:
                    # FP16 needs loss scaling to avoid underflow in gradients
                    self._grad_scaler.scale(loss).backward()
                    self._grad_scaler.unscale_(self.optimizer)
                else:
                    loss.backward()

                # clip_grad_norm_ also returns the pre-clip gradient norm, which lets us catch a
                # NaN/Inf gradient (e.g. from an unstable batch) before it corrupts the weights.
                # With BF16 there is no GradScaler to catch this automatically like there is for FP16.
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.cfg.clip_gradient_norm if self.cfg.clip_gradient_norm is not None else 1e9)
                step_ok = bool(torch.isfinite(grad_norm))

                if step_ok:
                    if self._grad_scaler is not None:
                        self._grad_scaler.step(self.optimizer)
                        self._grad_scaler.update()
                    else:
                        self.optimizer.step()

            if step_ok:
                nan_count = 0
            else:
                nan_count += 1
                if nan_count > self._allow_subsequent_nan_losses:
                    raise RuntimeError(f"Loss/gradients were NaN for {nan_count} times in a row. Stopped training.")
                LOGGER.warning(f"Loss or gradients are NaN; ignoring step. (#{nan_count}/{self._allow_subsequent_nan_losses})")

            pbar.set_postfix_str(f"Loss: {loss.item():.4f}")

            self.experiment_logger.log_step(**{k: v.item() for k, v in all_losses.items()})
    def _set_random_seeds(self):
        if self.cfg.seed is None:
            self.cfg.seed = int(np.random.uniform(low=0, high=1e6))

        # fix random seeds for various packages
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.cuda.manual_seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)

    def _set_device(self):
        if self.cfg.device is not None:
            if self.cfg.device.startswith("cuda"):
                gpu_id = int(self.cfg.device.split(':')[-1])
                if gpu_id > torch.cuda.device_count():
                    raise RuntimeError(f"This machine does not have GPU #{gpu_id} ")
                else:
                    self.device = torch.device(self.cfg.device)
            elif self.cfg.device == "mps":
                if torch.backends.mps.is_available():
                    self.device = torch.device("mps")
                else:
                    raise RuntimeError("MPS device is not available.")
            else:
                self.device = torch.device("cpu")
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        LOGGER.info(f"### Device {self.device} will be used for training")

        if self.device.type == 'cuda':
            # lets cuDNN benchmark different kernel implementations and pick the fastest one for our input sizes.
            # pays off quickly since seq_length and hidden_size are fixed for the whole training run.
            torch.backends.cudnn.benchmark = True

            # BF16 keeps the same dynamic range as FP32 (no underflow risk), so no GradScaler needed.
            # Fall back to FP16 + GradScaler on older GPUs that don't support BF16 (pre-Ampere).
            if torch.cuda.is_bf16_supported():
                self._use_amp = True
                self._amp_dtype = torch.bfloat16
                self._grad_scaler = None
                LOGGER.info("AMP enabled with BF16 (no gradient scaling needed)")
            else:
                self._use_amp = True
                self._amp_dtype = torch.float16
                self._grad_scaler = torch.cuda.amp.GradScaler()
                LOGGER.info("AMP enabled with FP16 + GradScaler")
        else:
            self._use_amp = False
            self._amp_dtype = None
            self._grad_scaler = None

    def _create_folder_structure(self):
        if self.cfg.is_continue_training:
            # Train directly inside the original run directory instead of a nested
            # continue_training_from_epochXXX/ subfolder. This keeps checkpoints, optimizer
            # states, and output.log in one place across every resume — evaluation
            # (evaluation/tester.py) only looks for model_epoch*.pt in the top-level run dir.
            self.cfg.base_run_dir = self.cfg.run_dir
            self.cfg.train_dir = self.cfg.run_dir / "train_data"
            self.cfg.train_dir.mkdir(parents=True, exist_ok=True)
            if self.cfg.log_n_figures is not None:
                self.cfg.img_log_dir = self.cfg.run_dir / "img_log"
                self.cfg.img_log_dir.mkdir(parents=True, exist_ok=True)
            return

        # create as new folder structure
        now = datetime.now()
        day = f"{now.day}".zfill(2)
        month = f"{now.month}".zfill(2)
        hour = f"{now.hour}".zfill(2)
        minute = f"{now.minute}".zfill(2)
        second = f"{now.second}".zfill(2)
        run_name = f'{self.cfg.experiment_name}_{day}{month}_{hour}{minute}{second}'

        # if no directory for the runs is specified, a 'runs' folder will be created in the current working dir
        if self.cfg.run_dir is None:
            self.cfg.run_dir = Path().cwd() / "runs" / run_name
        else:
            self.cfg.run_dir = self.cfg.run_dir / run_name

        # create folder + necessary subfolder
        if not self.cfg.run_dir.is_dir():
            self.cfg.train_dir = self.cfg.run_dir / "train_data"
            self.cfg.train_dir.mkdir(parents=True)
        else:
            raise RuntimeError(f"There is already a folder at {self.cfg.run_dir}")
        if self.cfg.log_n_figures is not None:
            self.cfg.img_log_dir = self.cfg.run_dir / "img_log"
            self.cfg.img_log_dir.mkdir(parents=True)
