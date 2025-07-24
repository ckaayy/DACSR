from .SASRecT import Model
from .SASRecT1 import Model_1
from .SASRecTW import SASRecWModel
from .SIGMA import SIGMA
from .LinRec import LinRec
# from .ColdNas import ColdNas
MODELS = {
   
    Model.code(): Model,
    Model_1.code(): Model_1,
    SASRecWModel.code():SASRecWModel,
    SIGMA.code(): SIGMA,
    LinRec.code(): LinRec,
    # ColdNas.code(): ColdNas,
    
}


def model_factory(args):
    if args.model_code == 'SIGMA':
        # explicitly pass the right fields
        m = SIGMA(
            num_items    = args.num_items + 1,
            hidden_size  = args.hidden_units,
            num_layers   = args.num_layers,
            dropout_prob = args.dropout_rate,
            loss_type    = args.loss_type,
            d_state      = args.d_state,
            d_conv       = args.d_conv,
            expand       = args.expand,
        )
        return m.to(args.device)
    else:
        cls = MODELS[args.model_code]
        return cls(args)