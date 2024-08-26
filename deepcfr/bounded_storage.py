import torch
import numpy as np


class BoundedStorage:
    def __init__(self, max_size):
        self.max_size = max_size
        self.storage = [None] * max_size
        self.current_idx = 0
        self.current_len = 0

    def get_storage(self):
        return self.storage[: self.current_len]

    def add_all(self, items):
        if len(items) > self.max_size:
            raise ValueError("Too many items to add")

        if len(items) + self.current_len <= self.max_size:
            self.current_len += len(items)
            self.storage[self.current_idx : self.current_idx + len(items)] = items
            self.current_idx = (self.current_idx + len(items)) % self.max_size
            return

        if self.current_len < self.max_size:
            first_part = self.max_size - self.current_len
            self.storage[self.current_idx :] = items[:first_part]
            self.current_len = self.max_size
            self.current_idx = 0
            items = items[first_part:]

        if self.current_idx + len(items) <= self.max_size:
            self.storage[self.current_idx : self.current_idx + len(items)] = items
            self.current_idx = (self.current_idx + len(items)) % self.max_size
        else:
            first_part = self.max_size - self.current_idx
            self.storage[self.current_idx :] = items[:first_part]
            self.storage[: len(items) - first_part] = items[first_part:]
            self.current_idx = len(items) - first_part


class GPUBoundedStorage:
    def __init__(self, max_size, target_size=4):
        self.max_size = max_size
        self.current_len = 0
        self.current_idx = 0

        self.obs = {
            "board_and_hand": torch.zeros(
                (max_size, 21), device="cuda", dtype=torch.long
            ),
            "stage": torch.zeros(max_size, device="cuda", dtype=torch.long),
            "first_to_act_next_stage": torch.zeros(
                max_size, device="cuda", dtype=torch.long
            ),
            "bets_and_stacks": torch.zeros((max_size, 8), device="cuda"),
        }

        self.ts = torch.zeros((max_size, 1), device="cuda")
        self.values = torch.zeros((max_size, target_size), device="cuda")

    def get_storage(self):
        if self.current_len == self.max_size:
            return self.obs, self.ts, self.values
        # otherwise slice it to the current length
        ret_obs = {k: v[: self.current_len] for k, v in self.obs.items()}
        return ret_obs, self.ts[: self.current_len], self.values[: self.current_len]

    def add_all(self, items):
        if not items:
            return

        obses = {
            k: torch.tensor(
                [item[0][k] for item in items], device="cuda", dtype=torch.long
            )
            for k in [
                "board_and_hand",
                "stage",
                "first_to_act_next_stage",
                "bets_and_stacks",
            ]
        }

        ts = torch.tensor([item[1] for item in items], device="cuda")
        values = torch.tensor(np.array([item[2] for item in items]), device="cuda")

        if self.current_len + len(items) <= self.max_size:
            start_idx = self.current_len
            end_idx = self.current_len + len(items)
            self.current_len += len(items)
            for k, v in obses.items():
                self.obs[k][start_idx:end_idx] = v
            self.ts[start_idx:end_idx] = ts[..., None]
            self.values[start_idx:end_idx] = values
            return

        if self.current_len < self.max_size:
            first_part = self.max_size - self.current_len
            for k, v in obses.items():
                self.obs[k][self.current_len :] = v[:first_part]
            self.ts[self.current_len :] = ts[:first_part][..., None]
            self.values[self.current_len :] = values[:first_part]
            self.current_len = self.max_size

            for k, v in obses.items():
                self.obs[k][: len(items) - first_part] = v[first_part:]
            self.ts[: len(items) - first_part] = ts[first_part:][..., None]
            self.values[: len(items) - first_part] = values[first_part:]
            self.current_idx = len(items) - first_part
            return

        if self.current_idx + len(items) <= self.max_size:
            for k, v in obses.items():
                self.obs[k][self.current_idx : self.current_idx + len(items)] = v
            self.ts[self.current_idx : self.current_idx + len(items)] = ts[..., None]
            self.values[self.current_idx : self.current_idx + len(items)] = values
            self.current_idx = (self.current_idx + len(items)) % self.max_size
            return

        first_part = self.max_size - self.current_idx
        for k, v in obses.items():
            self.obs[k][self.current_idx :] = v[:first_part]
        self.ts[self.current_idx :] = ts[:first_part][..., None]
        self.values[self.current_idx :] = values[:first_part]
        self.current_idx = 0

        for k, v in obses.items():
            self.obs[k][: len(items) - first_part] = v[first_part:]
        self.ts[: len(items) - first_part] = ts[first_part:][..., None]
        self.values[: len(items) - first_part] = values[first_part:]
        self.current_idx = len(items) - first_part


if __name__ == "__main__":
    storage = BoundedStorage(5)
    storage.add_all([1, 2, 3])
    assert storage.get_storage() == [1, 2, 3]
    storage.add_all([4, 5, 6])
    assert storage.get_storage() == [6, 2, 3, 4, 5]
    storage.add_all([0])
    assert storage.get_storage() == [6, 0, 3, 4, 5]
    storage.add_all([7, 7, 7, 7])
    assert storage.get_storage() == [7, 0, 7, 7, 7]
    storage.add_all([8, 8])
    assert storage.get_storage() == [7, 8, 8, 7, 7]
    try:
        storage.add_all([9, 9, 9, 9, 9, 9])
    except ValueError:
        pass
    else:
        raise AssertionError("ValueError not caught")
    storage.add_all([1, 2, 3, 4, 5])
    assert storage.get_storage() == [3, 4, 5, 1, 2]
    print("All tests passed")
