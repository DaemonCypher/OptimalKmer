import json
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import seaborn as sns

def load_metadata_files(base_dir):
    """Load metadata from all k-value directories."""
    metadata_by_k = {}
    
    # List all k-value directories
    k_dirs = [d for d in os.listdir(base_dir) if d.startswith('k')]
    
    for k_dir in k_dirs:
        k_value = int(k_dir[1:])  # Extract k value from directory name
        k_path = os.path.join(base_dir, k_dir)
        metadata_files = [f for f in os.listdir(k_path) if f.startswith('metadata_')]
        
        metadata_by_k[k_value] = {}
        for metadata_file in metadata_files:
            level = metadata_file.split('_')[1].split('.')[0]  # Extract taxonomic level
            with open(os.path.join(k_path, metadata_file), 'r') as f:
                metadata_by_k[k_value][level] = json.load(f)
    
    return metadata_by_k

def plot_level_performance(metadata_by_k, level, output_file=None):
    """Create comprehensive plots for a specific taxonomic level across k values."""
    level_names = {
        'd': 'Domain',
        'p': 'Phylum',
        'c': 'Class',
        'o': 'Order',
        'f': 'Family',
        'g': 'Genus',
        's': 'Species'
    }
    
    level_name = level_names.get(level, level)
    k_values = sorted(metadata_by_k.keys())
    
    # Create figure with GridSpec
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 2, figure=fig)
    
    # 1. Training History Plot (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    for k in k_values:
        history = metadata_by_k[k][level]['training_history']
        epochs = range(1, len(history['accuracy']) + 1)
        
        ax1.plot(epochs, history['accuracy'], 
                label=f'k={k} (train)', 
                linestyle='-', 
                alpha=0.7)
        ax1.plot(epochs, history['val_accuracy'], 
                label=f'k={k} (val)', 
                linestyle='--', 
                alpha=0.7)
    
    ax1.set_title(f'{level_name} Level - Training History')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Test Accuracy vs K-value (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    test_accuracies = [metadata_by_k[k][level]['test_accuracy'] for k in k_values]
    
    ax2.plot(k_values, test_accuracies, 'o-', linewidth=2, markersize=10)
    for k, acc in zip(k_values, test_accuracies):
        ax2.annotate(f'{acc:.4f}', 
                    (k, acc), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')
    
    ax2.set_title(f'{level_name} Level - Test Accuracy vs k-value')
    ax2.set_xlabel('k-value')
    ax2.set_ylabel('Test Accuracy')
    ax2.grid(True, alpha=0.3)
    
    # 3. Loss Curves (bottom left)
    ax3 = fig.add_subplot(gs[1, 0])
    for k in k_values:
        history = metadata_by_k[k][level]['training_history']
        epochs = range(1, len(history['loss']) + 1)
        
        ax3.plot(epochs, history['loss'], 
                label=f'k={k} (train)', 
                linestyle='-', 
                alpha=0.7)
        ax3.plot(epochs, history['val_loss'], 
                label=f'k={k} (val)', 
                linestyle='--', 
                alpha=0.7)
    
    ax3.set_title(f'{level_name} Level - Loss History')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Loss')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. Performance Metrics Summary (bottom right)
    ax4 = fig.add_subplot(gs[1, 1])
    metrics_data = []
    for k in k_values:
        metrics_data.append({
            'k-value': k,
            'Test Accuracy': metadata_by_k[k][level]['test_accuracy'],
            'Test Loss': metadata_by_k[k][level]['test_loss'],
            'Classes': metadata_by_k[k][level]['classes'],
            'Features': metadata_by_k[k][level]['features']
        })
    
    # Create table
    cell_text = []
    for metrics in metrics_data:
        cell_text.append([
            f"{metrics['k-value']}",
            f"{metrics['Test Accuracy']:.4f}",
            f"{metrics['Test Loss']:.4f}",
            f"{metrics['Classes']}",
            f"{metrics['Features']}"
        ])
    
    table = ax4.table(cellText=cell_text,
                     colLabels=['k-value', 'Test Acc', 'Test Loss', 'Classes', 'Features'],
                     loc='center',
                     cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    ax4.set_title(f'{level_name} Level - Performance Metrics Summary')
    ax4.axis('off')
    
    # Adjust layout and save
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {output_file}")
    plt.show()

# Example usage
if __name__ == "__main__":
    # Load metadata from all k-value directories
    base_dir = "trained_models"  # Adjust this path to your model directory
    metadata_by_k = load_metadata_files(base_dir)
    
    # Plot for each taxonomic level
    for level in ['d','p', 'c', 'o', 'f', 'g', 's']:
        output_file = f"performance_plot_{level}.png"
        try:
            plot_level_performance(metadata_by_k, level, output_file)
        except KeyError as e:
            print(f"Skipping level {level}: Data not found")