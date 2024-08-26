import torch


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

        self.obs = {
            "board_and_hand": torch.zeros(
                (max_size, 21), device="cuda", dtype=torch.long
            ),
            "stage": torch.zeros((max_size, 1), device="cuda", dtype=torch.long),
            "first_to_act_next_stage": torch.zeros(
                (max_size, 1), device="cuda", dtype=torch.long
            ),
            "bets_and_stacks": torch.zeros((max_size, 8), device="cuda"),
        }

        self.ts = torch.zeros((max_size, 1), device="cuda")
        self.values = torch.zeros((max_size, target_size), device="cuda")

    def get_storage(self):
        ret_obs = {k: v[: self.current_len] for k, v in self.obs.items()}
        return ret_obs, self.ts[: self.current_len], self.values[: self.current_len]

    def add_all(self, items):
        for obs, ts, values in items:
            idx = None
            if self.current_len < self.max_size:
                idx = self.current_len
                self.current_len += 1
            else:
                idx = torch.randint(0, self.max_size, (1,)).item()
            self.obs["board_and_hand"][idx] = torch.tensor(
                obs["board_and_hand"], device="cuda", dtype=torch.int32
            )
            self.obs["stage"][idx] = torch.tensor(obs["stage"], device="cuda")
            self.obs["first_to_act_next_stage"][idx] = torch.tensor(
                obs["first_to_act_next_stage"], device="cuda"
            )
            self.obs["bets_and_stacks"][idx] = torch.tensor(
                obs["bets_and_stacks"], device="cuda"
            )
            self.ts[idx] = torch.tensor(ts, device="cuda")
            self.values[idx] = torch.tensor(values, device="cuda")


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
