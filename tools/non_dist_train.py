# Copyright (c) CAIRI AI Lab. All rights reserved

import os.path as osp
import warnings
warnings.filterwarnings('ignore')

from simvp.api import NodDistExperiment
from simvp.utils import create_parser, load_config, update_config
import ipdb

try:
    import nni
    has_nni = True
except ImportError: 
    has_nni = False


if __name__ == '__main__':
    args = create_parser().parse_args()
    config = args.__dict__

    if has_nni:
        tuner_params = nni.get_next_parameter()
        config.update(tuner_params)

    # import ipdb
    # ipdb.set_trace()
    cfg_path = osp.join('./configs', args.dataname, f'{args.method}.py') \
        if args.config_file is None else args.config_file
    
    config = update_config(config, load_config(cfg_path),
                           exclude_keys=['batch_size', 'val_batch_size'])

    exp = NodDistExperiment(args)
    
    # ipdb.set_trace()
    if args.is_train:
        print('>'*35 + ' training ' + '<'*35)
        exp.train()
    else:
        print('>'*35 + ' testing  ' + '<'*35)
        mse = exp.test()
        if has_nni:
            nni.report_final_result(mse)
