
from .CE_Coral import CETrainer
from .CE_W import CEWTrainer

TRAINERS = {
    CEWTrainer.code():CEWTrainer,
    CETrainer.code():CETrainer,
}

def trainer_factory(args, model, train_loader_source, train_loader_target, train_loader_combine, val_loader, test_loader, export_root):
    trainer = TRAINERS[args.trainer_code]
    return trainer(args, model, train_loader_source, train_loader_target, train_loader_combine, val_loader, test_loader, export_root)
