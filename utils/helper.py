import torch
import os
from typing import Literal
from typing import NamedTuple
import math

def save_checkpoint(state, model, checkpoint_dir, epoch):
    # Remove the previous checkpoint only if the epoch is greater than 0
    if epoch > 0:
        # Remove the previous checkpoint
        previous_checkpoint = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch - 1}.pth')
        if os.path.exists(previous_checkpoint):
            try:
                os.remove(previous_checkpoint)
                print(f"Previous checkpoint {previous_checkpoint} removed.")
            except Exception as e:
                print(f"Error removing previous checkpoint: {e}")
        else:
            print(f"No previous checkpoint found at {previous_checkpoint}")
        
        # Remove the previous scripted model
        previous_scripted_model = os.path.join(checkpoint_dir, f'scripted_model_epoch_{epoch - 1}.pt')
        if os.path.exists(previous_scripted_model):
            try:
                os.remove(previous_scripted_model)
                print(f"Previous scripted model {previous_scripted_model} removed.")
            except Exception as e:
                print(f"Error removing previous scripted model: {e}")
        else:
            print(f"No previous scripted model found at {previous_scripted_model}")
    
    # Save the new checkpoint
    filename = f'checkpoint_epoch_{epoch}.pth'
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save(state, checkpoint_path)

    # Script the model and save it as part of the checkpointing process
    # Check if the model is wrapped in DataParallel
    if isinstance(model, torch.nn.DataParallel):
        model = model.module  # Unwrap DataParallel to get the original model

    # Save the new scripted model
    scripted_model_path = os.path.join(checkpoint_dir, f'scripted_model_epoch_{epoch}.pt')
    scripted_model = torch.jit.script(model)
    torch.jit.save(scripted_model, scripted_model_path)

    print(f"Checkpoint saved at {checkpoint_path}, scripted model saved at {scripted_model_path}")



def load_checkpoint(checkpoint_path, model, optimizer):
    # Load the optimizer state and other metadata
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Resuming training from epoch {epoch}")
    return epoch
