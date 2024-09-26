import torch
import wandb
import hydra
from data import get_train_test
from invariance_test import test_invariance
from models import get_model
from train_and_evaluate import train_and_evaluate


@hydra.main(config_path=".", config_name="cfg", version_base=None)
def main(cfg):
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(cfg.seed)
    train_loader, test_loader = get_train_test(train_size=cfg.data.train_size,
                                               test_size=cfg.data.test_size,
                                               set_size=cfg.data.set_size,
                                               data_dim=cfg.data.data_dim,
                                               batch_size= cfg.training.batch_size,
                                               device=cfg.device)

    input_dim = cfg.data.data_dim  if cfg.architecture.model_type=='InvariantLinearlNetwork' else cfg.data.data_dim * cfg.data.set_size
    model = get_model(model_type=cfg.architecture.model_type,
                      input_dim=input_dim,
                      hidden_dim=cfg.architecture.hidden_dim,
                      output_dim=cfg.data.output_dim,
                      num_layers=cfg.architecture.num_layers,
                      device=cfg.device)

    if cfg.wandb.log:
        wandb.init(
            settings=wandb.Settings(start_method="thread"),
            project=cfg.wandb.project_name,
            config=dict(cfg),
        )
    if cfg.run_type=="train":
        train_and_evaluate(train_loader=train_loader,
                           test_loader=test_loader,
                           model=model,
                           device=cfg.device,
                           lr=cfg.training.lr,
                           epochs=cfg.training.epochs,
                           model_type=cfg.architecture.model_type,
                           log_wandb= cfg.wandb.log)

    elif cfg.run_type == 'test':
        test_invariance(model=model, set_size=cfg.data.set_size, data_dim=cfg.data.data_dim)


if __name__ == '__main__':
    main()
