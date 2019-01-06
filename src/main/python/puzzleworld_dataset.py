from torch.utils.data import Dataset

class PuzzleWorldDataset(Dataset):
    def __init__(self, size, seq_len, random_seed=0, feature_type="PREDEFINED"):
        super(ModelDataSet, self).__init__()
        self.my_puzzle_world = bs.PuzzleWorld(feature_type=feature_type)
        self.feature_type = feature_type
        self.size = size
        self.seq_len = seq_len
        seed(random_seed)

    def __len__(self):
        return self.size

    def _random_force(self):
        force = np.array([random() - 0.5, random() - 0.5], dtype=np.float32)
        return force

    def __getitem__(self, idx):
        self.my_block_sys.reset()
        force = np.zeros([seq_len, 2], dtype=np.float32)
        if feature_type == "PREDEFINED":
            obsv = np.zeros([seq_len, TOTAL_FEATURE_COUNT], dtype=np.float32)
        else:
            raise NotImplementedError
        # Generate seq_len + 1 quad frames since minimum 2 (input + label).
        for seq_idx in range(self.seq_len + 1):
            force[seq_idx] = self._random_force()
            # Collect 4 frames
            obsv[seq_idx, :] = self.my_block_sys.step(
                force_item[seq_idx, 0], force[seq_idx, 1]
            )
        return (force, obsv)
