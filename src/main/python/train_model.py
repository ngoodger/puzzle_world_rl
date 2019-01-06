import torch.distributed as dist
import puzzle_world_dataset
import os

TRAINING_ITERATIONS = 100000000
TRAINING_TIME = timedelta(minutes=20)
SEQ_LEN = 6
SAVE_INTERVAL = 1000
PRINT_INTERVAL = 100
GRU_DEPTH = 2
GRU_HIDDEN_SIZE = 32

def objective(model_type="RECURRENT", learning_rate, batch_size, world_size, time_limit=TRAINING_TIME):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rank = dist.get_rank() if world_size > 1 else 0
    samples_dataset = puzzle_world_dataset.ModelDataSet(TRAINING_ITERATIONS, SEQ_LEN, rank)
    dataloader = DataLoader(
        samples_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    if model_type == "RECURRENT":
        my_model = model.PredefinedFeatureModelGRU(
            gru_depth=2,
            gru_hidden_size=2,
            device=device,
        ).to(device)
    else:
        raise NotImplementedError
    trainer = model.PredefinedFeatureModelTrainer(
        learning_rate=learning_rate,
        model=my_model,
    )
    for batch_idx, data in enumerate(dataloader):
        force, obsv = data
        force_device = torch.tensor(force, device=device)
        obsv_device = torch.tensor(obsv, device=device)
        mean_loss = trainer.train(batch_data)
            
