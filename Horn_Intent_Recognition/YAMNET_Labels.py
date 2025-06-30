import yamnet
import params

# Load class names
class_names = yamnet.class_names('yamnet_class_map.csv')

# List all labels containing 'horn'
print("Available 'horn' labels:\n")
for i, name in enumerate(class_names):
    if 'horn' in name.lower():
        print(f"Index: {i}, Name: {name}")
