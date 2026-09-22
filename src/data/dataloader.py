import torch


class SimulationDataloader:
    def __init__(self, dataset, batch_size=32, shuffle=False, device='cpu', drop_last=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.drop_last = shuffle if drop_last is None else drop_last

        if hasattr(self.dataset, 'to') and callable(self.dataset.to):
            self.dataset.to(self.device)

    def __len__(self):
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        dataset_size = len(self.dataset)
        indices = torch.randperm(dataset_size) if self.shuffle else torch.arange(dataset_size)

        for start_idx in range(0, dataset_size, self.batch_size):
            batch_indices = indices[start_idx : start_idx + self.batch_size]

            if len(batch_indices) == 0 or (len(batch_indices) < self.batch_size and self.drop_last):
                break

            states = self.dataset.states[batch_indices].to(self.device)
            actions = self.dataset.actions[batch_indices].to(self.device)

            b, t = states.shape[:2]
            images = self.dataset.draw_states(states.reshape(b * t, -1))
            _, c, h, w = images.shape
            yield {
                'images': images.reshape(b, t, c, h, w),
                'actions': actions,
                'states': states,
            }
