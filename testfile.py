import torch

# Load the file
data = torch.load(r"C:\Users\SAKETH\Downloads\EaSe\data\tables.pkl")

# Print the type of the loaded object
print(type(data))

# Optional: explore keys if it's a dictionary
if isinstance(data, dict):
    print("Keys:", list(data.keys()))

    # For example, inspect one ite
    first_key = list(data.keys())[0]
    print(f"Shape of first entry ({first_key}):", data[first_key].shape)
else:
    print("Loaded object:", data)
