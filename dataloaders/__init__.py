from datasets import dataset_factory

from .SASRec import SASRecDataloader

DATALOADERS = {
    
    SASRecDataloader.code(): SASRecDataloader,
    
}


def dataloader_factory(args):
    dataset = dataset_factory(args)
    dataloader = DATALOADERS[args.dataloader_code]
    dataloader = dataloader(args, dataset)
    train_source, train_target, train_combine, val, test = dataloader.get_pytorch_dataloaders()
    if args.model_code == 'SIGMA':
        # SIGMA only needs the “combined” view of ALL your sequences
        # reuse whatever loader you already make for train_loader_combine:
        return train_combine,train_combine,train_combine,val, test
    return train_source,train_target, train_combine, val, test
