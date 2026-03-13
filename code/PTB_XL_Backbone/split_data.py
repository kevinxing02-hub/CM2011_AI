import torch

def prepare_folded_ptb():
    # Load original 10s data
    train_data = torch.load('used_data/data_basic_ptb/ptbxl_train.pt', weights_only=False)
    test_data = torch.load('used_data/data_basic_ptb/ptbxl_test.pt', weights_only=False)

    for name, data in [("train", train_data), ("test", test_data)]:
        X, y = data['X'], data['y']
        
        # Split [N, Leads, 1000] -> four [N, Leads, 250]
        chunks = torch.split(X, 250, dim=2)
        X_folded = torch.cat(chunks, dim=0)
        
        # Repeat labels 4 times [N*4, Classes]
        y_folded = torch.cat([y] * 4, dim=0)
        
        # Save as a NEW file so you don't overwrite your 10s data
        torch.save({
            'X': X_folded,
            'y': y_folded,
            'classes': data['classes']
        }, f'used_data/data_basic_ptb/ptbxl_{name}_250.pt')
        print(f"Folded {name} set: {X_folded.shape}")

if __name__ == "__main__":
    prepare_folded_ptb()