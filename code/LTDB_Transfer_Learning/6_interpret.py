import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
from model import LeadAgnosticTransformer

# Path config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Want to visualize how the model reacted to ptbxl and ltdb. So we take the best ptbxl model and the best ltdb model. 
TASKS = {
    "PTBXL_Base": {
        "model_path": os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'results', 'ptbxl_v1', 'best_model.pt')),
        "data_path": os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'used_data', 'data_ptb', 'ptbxl_test.pt')),
        "num_classes": 5, 
        "folder": "ptbxl_12lead_analysis"
    },
    "LTDB_Transfer": {
        "model_path": os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'results', 'ltdb_v1', 'best_transfer_model.pt')),
        "data_path": os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'used_data', 'data_ltdb', 'ltdb_test.pt')),
        "num_classes": 6, 
        "folder": "ltdb_2lead_analysis"
    }
}

OUTPUT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'results', 'interpretability_final'))
  # Reshape [Batch, Leads, Samples] To [Batch * Leads, 1, Samples]. Each lead gets analysed seperately.


# Get the attention weights and aggregate over all layers. 
def get_last_layer_attention(model, x_tensor):
    model.eval()
    with torch.no_grad():
        b, l, t = x_tensor.shape
        
        x = x_tensor.reshape(b * l, 1, t)
        x = model.cnn(x) # CNN outputs [Batch * leads, Features, 1]
        x = x.view(b, l, -1) #Fold the shape back to [batch, leads, Features] before sending to transformer

        # Forward through all layers except last
        for layer in model.transformer.layers[:-1]:
            x = layer(x)

        # Get attention from LAST layer only
        last_layer = model.transformer.layers[-1]
        _, attn_weights = last_layer.self_attn(
            x, x, x,
            need_weights=True,
            average_attn_weights=True
        )

    return attn_weights.cpu().numpy()  # [B, L, L]

# SHAP Waveform

def run_lead_shap(model, X_batch, classes, out_dir):
    print(f"Generating SHAP plots in {out_dir}...")
    device = next(model.parameters()).device
    
    baseline = X[10:30].to(device)  # 20 Samples to use as a baseline
    test_samples = X[:3].to(device) # Same 3 samples used in the attention maps
    
    explainer = shap.GradientExplainer(model, baseline)
    shap_vals = explainer.shap_values(test_samples)


    all_shap_vals = np.concatenate([v for v in shap_vals]) if isinstance(shap_vals, list) else shap_vals
    v_limit = np.percentile(np.abs(all_shap_vals), 99)

    # Save the scale in seperate picture so it doesnt cover waveform
    save_standalone_colorbar(v_limit, 'RdBu_r', os.path.join(out_dir, "shap_colorbar_scale.png"))

    for i in range(len(test_samples)):
        with torch.no_grad():
            output = model(test_samples[i:i+1])
            pred_idx = torch.argmax(output, dim=1).item()
        
        current_shap_sample = shap_vals[pred_idx][i] if isinstance(shap_vals, list) else shap_vals[i, :, :, pred_idx]

        num_leads = test_samples.shape[1]
        
        if num_leads == 12:
            # 12-lead Grid (6 rows, 2 columns)
            fig, axes = plt.subplots(6, 2, figsize=(14, 18), sharex=True)
            axes_flat = axes.flatten()
        else:
            # 2-lead Stack (2 rows, 1 column)
            fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
            axes_flat = axes.flatten()

        for lead_idx in range(num_leads):
            signal = test_samples[i, lead_idx, :].cpu().numpy().flatten()
            s_val = current_shap_sample[lead_idx].flatten()

            ax = axes_flat[lead_idx]
            ax.plot(signal, color='gray', alpha=0.3, lw=1) 
            
            # Make sure scatter plot is on top
            sc = ax.scatter(range(len(signal)), signal, c=s_val, 
                            cmap='RdBu_r', s=8, edgecolors='none', 
                            vmin=-v_limit, vmax=v_limit, zorder=3)
            
            ax.set_ylabel(f"Lead {lead_idx}", fontsize=9)
            if lead_idx == 0:
                ax.set_title(f"Sample {i} | Predicted: {classes[pred_idx]}", fontsize=12)


        # Save plots
        plt.savefig(os.path.join(out_dir, f"shap_full_leads_sample_{i}.png"))
        plt.savefig(os.path.join(out_dir, f"shap_full_leads_sample_{i}.png"))

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(out_dir, f"shap_full_leads_sample_{i}.png"))
        plt.close()



def lead_masking_importance(model, x_batch, device):
    model.eval()
    x_batch = x_batch.to(device)
    
    with torch.no_grad():
        # Original predictions
        original_output = model(x_batch)  # [B, C]
        original_probs = torch.sigmoid(original_output)

    b, l, t = x_batch.shape
    importance_scores = torch.zeros(b, l)

    for lead_idx in range(l):
        x_masked = x_batch.clone()
        
        # MASK: zero out one lead
        x_masked[:, lead_idx, :] = 0
        
        with torch.no_grad():
            masked_output = model(x_masked)
            masked_probs = torch.sigmoid(masked_output)
        
        # Measure change 
        diff = torch.abs(original_probs - masked_probs).mean(dim=1)  # [B]
        
        importance_scores[:, lead_idx] = diff.cpu()

    return importance_scores  # [B, L]


def plot_masking(importance_scores, sample_idx, out_path):
    scores = importance_scores[sample_idx].numpy()
    
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(scores)), scores)
    plt.xlabel("Lead Index")
    plt.ylabel("Importance (Prediction Change)")
    plt.title(f"Lead Masking Importance - Sample {sample_idx}")
    
    plt.savefig(out_path)
    plt.close()


def save_standalone_colorbar(v_limit, cmap_name, out_path):
    """Generates a separate PNG file containing only the SHAP color scale."""
    import matplotlib as mpl
    
    # Create a figure specifically for the colorbar
    fig = plt.figure(figsize=(2, 6))
    ax = fig.add_axes([0.3, 0.1, 0.15, 0.8]) # [left, bottom, width, height]

    # Define the normalization based on your SHAP limits
    norm = mpl.colors.Normalize(vmin=-v_limit, vmax=v_limit)
    
    cb = mpl.colorbar.ColorbarBase(ax, cmap=plt.get_cmap(cmap_name),
                                   norm=norm,
                                   orientation='vertical')
    
    cb.set_label('SHAP Value (Impact on Prediction)', rotation=270, labelpad=15, fontsize=10)
    
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Colorbar scale saved to: {out_path}")    


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    for name, cfg in TASKS.items():
        print(f"\n{'='*10} Processing: {name} {'='*10}")
        
        # Check if files exist
        if not os.path.exists(cfg["model_path"]) or not os.path.exists(cfg["data_path"]):
            print(f"  [!] Missing files for {name}. Skipping...")
            continue

        task_out = os.path.join(OUTPUT_ROOT, cfg["folder"])
        os.makedirs(task_out, exist_ok=True)

        # Load Data
        data = torch.load(cfg["data_path"], weights_only=False)
        X = data['X'].float()
        classes = data['classes']

        # Load Model
        model = LeadAgnosticTransformer(num_classes=cfg["num_classes"]).to(device)
        model.load_state_dict(torch.load(cfg["model_path"], map_location=device))
        model.eval()


        print(f"  -> Extracting Attention Rollout ({X.shape[1]} leads)...")
        
        # Grab a small batch 
        sample_batch = X[:3].to(device)
        attention_batch = get_last_layer_attention(model, sample_batch)
        
        # Plot attention map for each individual sample
        for i in range(len(sample_batch)):
            sample_attn = attention_batch[i] # Shape: [Leads, Leads]
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(sample_attn, annot=(X.shape[1] < 5), cmap='viridis')
            
            plt.title(f"Attention Last Layer: {name} - Sample {i} ({X.shape[1]} Leads)")
            plt.ylabel("Target Lead (Receiving Attention)")
            plt.xlabel("Source Lead (Giving Attention)")
            
            plt.savefig(os.path.join(task_out, f"lead_attention_sample_{i}.png"))
            plt.close()

        # Run shap for all 
        run_lead_shap(model, X, classes, task_out)


        print(f"  -> Running Lead Masking Importance...")

        # Use same samples as attention
        masking_batch = X[:3]

        importance_scores = lead_masking_importance(model, masking_batch, device)

        for i in range(len(masking_batch)):
            out_path = os.path.join(task_out, f"lead_masking_sample_{i}.png")
            plot_masking(importance_scores, i, out_path)

    print(f"\nDone! All results saved in: {OUTPUT_ROOT}")