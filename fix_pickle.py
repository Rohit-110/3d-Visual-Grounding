import torch
import pickle
import os

input_path = 'data/tables.pkl'  # your original file
output_path = 'data/fixed_tables.pkl'  # fixed output location

# This patched restore function correctly handles UntypedStorage
def patched_restore_location(storage, location):
    if isinstance(storage, torch.storage.UntypedStorage):
        return torch.UntypedStorage(storage.nbytes(), device="cpu")
    else:
        return storage

# Monkey-patch BEFORE loading
torch.serialization.default_restore_location = patched_restore_location

# Now load safely
with open(input_path, 'rb') as f:
    data = pickle.load(f)

# Recursively move any nested tensors to CPU (for extra safety)
def move_to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    elif isinstance(obj, dict):
        return {k: move_to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_cpu(x) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_cpu(x) for x in obj)
    else:
        return obj

fixed_data = move_to_cpu(data)

# Save the corrected file
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'wb') as f:
    pickle.dump(fixed_data,f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"✅ Successfully fixed and saved: {output_path}")