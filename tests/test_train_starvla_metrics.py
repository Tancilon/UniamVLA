from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace


class _FakeTensor:
    def __init__(self, value: float, numel: int = 1):
        self.value = value
        self._numel = numel

    def numel(self):
        return self._numel

    def detach(self):
        return self

    def float(self):
        return self

    def item(self):
        return self.value


def _install_train_starvla_stubs(monkeypatch):
    torch = types.ModuleType("torch")
    torch.bfloat16 = object()
    torch.is_tensor = lambda value: isinstance(value, _FakeTensor)

    class _Autocast:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    torch.autocast = _Autocast

    optim = types.ModuleType("torch.optim")
    optim.Optimizer = object
    lr_scheduler = types.ModuleType("torch.optim.lr_scheduler")
    lr_scheduler._LRScheduler = object
    optim.lr_scheduler = lr_scheduler
    torch.optim = optim

    dist = types.ModuleType("torch.distributed")
    dist.is_initialized = lambda: False
    dist.get_rank = lambda: 0
    dist.barrier = lambda: None
    dist.destroy_process_group = lambda: None
    torch.distributed = dist

    torch_utils = types.ModuleType("torch.utils")
    torch_utils_data = types.ModuleType("torch.utils.data")
    torch_utils_data.DataLoader = object
    torch_utils.data = torch_utils_data

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.optim", optim)
    monkeypatch.setitem(sys.modules, "torch.optim.lr_scheduler", lr_scheduler)
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)
    monkeypatch.setitem(sys.modules, "torch.utils", torch_utils)
    monkeypatch.setitem(sys.modules, "torch.utils.data", torch_utils_data)

    wandb = types.ModuleType("wandb")
    wandb.init = lambda *args, **kwargs: None
    wandb.log = lambda *args, **kwargs: None
    wandb.finish = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "wandb", wandb)

    accelerate = types.ModuleType("accelerate")

    class _Accelerator:
        state = "fake-state"
        dataloader_config = SimpleNamespace(dispatch_batches=True)
        is_main_process = True
        is_local_main_process = True
        sync_gradients = True

        def __init__(self, *args, **kwargs):
            pass

        def print(self, *args, **kwargs):
            pass

        def accumulate(self, _model):
            return _Autocast()

        def backward(self, loss):
            self.backward_loss = loss

        def clip_grad_norm_(self, *args, **kwargs):
            pass

    accelerate.Accelerator = _Accelerator
    accelerate.DeepSpeedPlugin = lambda *args, **kwargs: object()
    monkeypatch.setitem(sys.modules, "accelerate", accelerate)

    accelerate_logging = types.ModuleType("accelerate.logging")
    accelerate_logging.get_logger = lambda _name: SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "accelerate.logging", accelerate_logging)

    accelerate_utils = types.ModuleType("accelerate.utils")
    accelerate_utils.set_seed = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "accelerate.utils", accelerate_utils)

    omegaconf = types.ModuleType("omegaconf")
    omegaconf.OmegaConf = SimpleNamespace(
        load=lambda *args, **kwargs: None,
        from_dotlist=lambda *args, **kwargs: None,
        merge=lambda *args, **kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "omegaconf", omegaconf)

    transformers = types.ModuleType("transformers")
    transformers.AutoProcessor = SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: None,
    )
    transformers.get_scheduler = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    dataloader = types.ModuleType("starVLA.dataloader")
    dataloader.build_dataloader = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "starVLA.dataloader", dataloader)

    base_framework = types.ModuleType("starVLA.model.framework.base_framework")
    base_framework.build_framework = lambda *args, **kwargs: None
    monkeypatch.setitem(
        sys.modules,
        "starVLA.model.framework.base_framework",
        base_framework,
    )

    share_tools = types.ModuleType("starVLA.model.framework.share_tools")
    share_tools.apply_config_compat = lambda cfg: cfg
    monkeypatch.setitem(sys.modules, "starVLA.model.framework.share_tools", share_tools)

    config_tracker = types.ModuleType("starVLA.training.trainer_utils.config_tracker")
    config_tracker.AccessTrackedConfig = object
    config_tracker.wrap_config = lambda cfg: cfg
    monkeypatch.setitem(
        sys.modules,
        "starVLA.training.trainer_utils.config_tracker",
        config_tracker,
    )

    trainer_tools = types.ModuleType("starVLA.training.trainer_utils.trainer_tools")
    trainer_tools.TrainerUtils = object
    trainer_tools.build_param_lr_groups = lambda *args, **kwargs: []
    trainer_tools.normalize_dotlist_args = lambda args: args
    monkeypatch.setitem(
        sys.modules,
        "starVLA.training.trainer_utils.trainer_tools",
        trainer_tools,
    )


def test_train_step_logs_all_scalar_forward_metrics(monkeypatch):
    _install_train_starvla_stubs(monkeypatch)
    sys.modules.pop("starVLA.training.train_starvla", None)
    train_starvla = importlib.import_module("starVLA.training.train_starvla")

    class _Model:
        def forward(self, _batch, **_kwargs):
            return {
                "action_loss": _FakeTensor(10.0),
                "action_loss_l1": _FakeTensor(2.0),
                "recon_loss_weighted": _FakeTensor(0.5),
                "recon_loss_raw": 5.0,
                "future_loss_raw": 6,
                "not_scalar": _FakeTensor(99.0, numel=2),
                "not_numeric": "skip-me",
            }

        def parameters(self):
            return []

    optimizer = SimpleNamespace(
        zero_grad=lambda: None,
        step=lambda: None,
    )
    lr_scheduler = SimpleNamespace(step=lambda: None)
    accelerator = train_starvla.Accelerator()

    trainer = object.__new__(train_starvla.VLATrainer)
    trainer.model = _Model()
    trainer.optimizer = optimizer
    trainer.lr_scheduler = lr_scheduler
    trainer.accelerator = accelerator
    trainer.completed_steps = 0
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(gradient_clipping=None),
    )

    metrics = trainer._train_step(batch_vla={})

    assert accelerator.backward_loss.value == 10.0
    assert metrics == {
        "loss/total": 10.0,
        "loss/action_loss_l1": 2.0,
        "loss/recon_loss_weighted": 0.5,
        "loss/recon_loss_raw": 5.0,
        "loss/future_loss_raw": 6.0,
    }


def test_train_step_does_not_advance_scheduler_on_accumulation_microstep(monkeypatch):
    _install_train_starvla_stubs(monkeypatch)
    sys.modules.pop("starVLA.training.train_starvla", None)
    train_starvla = importlib.import_module("starVLA.training.train_starvla")

    class _Model:
        def forward(self, _batch, **_kwargs):
            return {"action_loss": _FakeTensor(10.0)}

        def parameters(self):
            return []

    class _Counter:
        def __init__(self):
            self.calls = 0

        def step(self):
            self.calls += 1

    optimizer_step = _Counter()
    scheduler_step = _Counter()
    optimizer = SimpleNamespace(
        zero_grad=lambda: None,
        step=optimizer_step.step,
    )
    lr_scheduler = SimpleNamespace(step=scheduler_step.step)
    accelerator = train_starvla.Accelerator()
    accelerator.sync_gradients = False

    trainer = object.__new__(train_starvla.VLATrainer)
    trainer.model = _Model()
    trainer.optimizer = optimizer
    trainer.lr_scheduler = lr_scheduler
    trainer.accelerator = accelerator
    trainer.completed_steps = 0
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(gradient_clipping=None),
    )

    trainer._train_step(batch_vla={})

    assert optimizer_step.calls == 1
    assert scheduler_step.calls == 0


def test_log_metrics_reports_epoch_from_global_batch_progress(monkeypatch):
    _install_train_starvla_stubs(monkeypatch)
    sys.modules.pop("starVLA.training.train_starvla", None)
    train_starvla = importlib.import_module("starVLA.training.train_starvla")

    logged = []
    monkeypatch.setattr(train_starvla.wandb, "log", lambda metrics, step: logged.append((dict(metrics), step)))

    class _Dataset:
        def __len__(self):
            return 1_046_099

    class _Dataloader:
        dataset = _Dataset()

        def __len__(self):
            return 16_345

    trainer = object.__new__(train_starvla.VLATrainer)
    trainer.completed_steps = 1600
    trainer.total_batch_size = 512
    trainer.vla_train_dataloader = _Dataloader()
    trainer.lr_scheduler = SimpleNamespace(get_last_lr=lambda: [1.9e-5])
    trainer.accelerator = train_starvla.Accelerator()
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(logging_frequency=50),
    )

    trainer._log_metrics({"loss/total": 0.1})

    assert len(logged) == 1
    metrics, step = logged[0]
    assert step == 1600
    assert metrics["epoch"] == round(1600 * 512 / 1_046_099, 2)
    assert metrics["epoch"] == 0.78


def test_log_metrics_skips_accumulation_microsteps(monkeypatch):
    _install_train_starvla_stubs(monkeypatch)
    sys.modules.pop("starVLA.training.train_starvla", None)
    train_starvla = importlib.import_module("starVLA.training.train_starvla")

    logged = []
    monkeypatch.setattr(train_starvla.wandb, "log", lambda metrics, step: logged.append((dict(metrics), step)))

    trainer = object.__new__(train_starvla.VLATrainer)
    trainer.completed_steps = 1600
    trainer.total_batch_size = 512
    trainer.vla_train_dataloader = SimpleNamespace(dataset=range(1_046_099))
    trainer.lr_scheduler = SimpleNamespace(get_last_lr=lambda: [1.9e-5])
    trainer.accelerator = train_starvla.Accelerator()
    trainer.accelerator.sync_gradients = False
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(logging_frequency=50),
    )

    trainer._log_metrics({"loss/total": 0.1})

    assert logged == []
