

def load_data(dataname, batch_size, val_batch_size, num_workers, data_root, **kwargs):
    pre_seq_length = kwargs.get('pre_seq_length', 10)
    aft_seq_length = kwargs.get('aft_seq_length', 10)
    train_data_paths = kwargs.get('train_data_paths', 'small_2000_10.npz')
    valid_data_paths = kwargs.get('valid_data_paths', 'small_2000_10.npz')
    test_data_paths = kwargs.get('test_data_paths', 'small_2000_10.npz')

    if dataname == 'radar':
        from .dataloader_radar import load_data
        return load_data(batch_size, val_batch_size, data_root, num_workers, pre_seq_length, aft_seq_length, train_data_paths, valid_data_paths, test_data_paths)
    else:
        raise ValueError(f'Dataname {dataname} is unsupported')
