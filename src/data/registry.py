from src.data.dataloader import SimulationDataloader
from src.data.dino import DinoDataset
from src.data.golf import GolfDataset
from src.data.pong import PongDataset


def registry_create_dataset(dataset_type='pong', **kwargs):
    if dataset_type == 'pong':
        return PongDataset(**kwargs)

    if dataset_type == 'dino':
        return DinoDataset(**kwargs)

    if dataset_type == 'golf':
        return GolfDataset(**kwargs)

    raise ValueError(
        f'Unknown dataset type: {dataset_type!r}. '
        f'Options: pong, dino, golf.'
    )


def registry_create_dataloader(dataset, batch_size=32, shuffle=False, device='cpu', drop_last=None):
    if hasattr(dataset, 'draw_states') and callable(dataset.draw_states):
        return SimulationDataloader(
            dataset, batch_size=batch_size, shuffle=shuffle, device=device, drop_last=drop_last,
        )
    raise TypeError(
        f'Dataset {type(dataset).__name__} has no draw_states; '
        'this release only supports simulation datasets.'
    )
