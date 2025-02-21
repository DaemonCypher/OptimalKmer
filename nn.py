import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import multiprocessing
from tensorflow.keras.mixed_precision import set_global_policy
import traceback
import os
import json
import pickle

# Configure CPU
def configure_cpu():
    """Configure TensorFlow for optimal CPU usage."""
    num_cores = multiprocessing.cpu_count()
    #print(f"Available CPU cores: {num_cores}")
    
    # Disable GPU
    tf.config.set_visible_devices([], 'GPU')
    
    # Set thread configuration
    tf.config.threading.set_intra_op_parallelism_threads(num_cores)
    tf.config.threading.set_inter_op_parallelism_threads(num_cores)
    
    # Use float32 precision
    set_global_policy('float32')
    
    #print("TensorFlow configured for CPU-only operation")

# Data Loading
def load_data():
    """Load and prepare taxonomy data."""
    df = pd.read_csv("species_taxonomy_data.tsv", sep='\t')
    df = df.drop(columns=['Confidence'])
    df = df.sample(frac=0.25, random_state=42)
    return df

def create_kmers(sequence, k=4):
    """Create k-mers from sequence."""
    kmers = []
    for i in range(len(sequence) - k + 1):
        kmers.append(sequence[i:i+k])
    return ' '.join(kmers)

def create_optimized_model(input_shape, num_classes, learning_rate=0.001):
    """Create model optimized for CPU computation."""
    hidden_units = min(512, max(128, num_classes * 4))
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        
        tf.keras.layers.Dense(
            hidden_units,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
            use_bias=True
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(
            hidden_units // 2,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
            use_bias=True
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Dense(
            hidden_units // 4,
            activation='relu',
            use_bias=True
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.1),
        
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=1.0,
        epsilon=1e-7
    )
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

class PlateauDetectionCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if epoch > 10:
            val_acc = logs.get('val_accuracy')
            if val_acc is not None:
                last_accuracies = self.model.history.history['val_accuracy'][-5:]
                if len(last_accuracies) >= 5:
                    std_dev = np.std(last_accuracies)
                    if std_dev < 0.0005:
                        print("\nStopping training: Accuracy has plateaued")
                        self.model.stop_training = True

def train_taxonomy_model(X, y, level_name, label_encoder, k):
    """Train model using CPU optimization with enhanced early stopping and model saving."""
    try:
        print(f"\nProcessing level {level_name} with k={k}")
        
        # Create directory for saving models if it doesn't exist
        save_dir = f"trained_models/k{k}"
        os.makedirs(save_dir, exist_ok=True)
        
        # Handle class imbalance
        unique, counts = np.unique(y, return_counts=True)
        min_samples = np.min(counts)
        
        if min_samples < 2:
            print(f"Handling rare classes for level {level_name}")
            valid_classes = unique[counts >= 3]
            mask = np.isin(y, valid_classes)
            X = X[mask]
            y = y[mask]
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Calculate class weights
        n_classes = len(np.unique(y))
        total_samples = len(y_train)
        class_weights = dict(zip(
            range(n_classes),
            [total_samples / (n_classes * np.sum(y_train == i)) for i in range(n_classes)]
        ))
        
        # Create model
        model = create_optimized_model(
            X.shape[1],
            n_classes,
            learning_rate=0.0005
        )
        
        # Model checkpoint callback for saving best model
        model_path = os.path.join(save_dir, f"model_{level_name}.keras")
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        )
        
        # Enhanced callbacks for better stopping criteria
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True,
                min_delta=0.0005,
                mode='max',
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.5,
                patience=3,
                min_lr=0.00001,
                min_delta=0.0005,
                mode='max',
                verbose=1
            ),
            checkpoint_callback,
            tf.keras.callbacks.TerminateOnNaN(),
            PlateauDetectionCallback()
        ]
        
        # Train with CPU optimization
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=128,  # Increased from 64
            validation_split=0.2,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        # Evaluate
        test_loss, test_accuracy = model.evaluate(
            X_test, 
            y_test, 
            batch_size=1024,
            verbose=1
        )
        
        print(f"{level_name} Level - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        
        # Save model metadata
        metadata = {
            'level_name': level_name,
            'k_value': k,
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'features': X.shape[1],
            'classes': n_classes,
            'training_history': history.history,
            'class_weights': class_weights
        }
        
        metadata_path = os.path.join(save_dir, f"metadata_{level_name}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4, default=str)
            
        # Save label encoder
        le_path = os.path.join(save_dir, f"label_encoder_{level_name}.pkl")
        with open(le_path, 'wb') as f:
            pickle.dump(label_encoder, f)
        
        print(f"Model and metadata saved in {save_dir}")
        
        return {
            'model': model,
            'history': history,
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'label_encoder': le,
            'features': X.shape[1],
            'classes': n_classes
        }
        
    except Exception as e:
        print(f"Error processing level {level_name} for k={k}: {str(e)}")
        traceback.print_exc()
        return None
    
def plot_kmer_comparison(results, taxonomic_levels=['p', 'c', 'o', 'f', 'g', 's']):
    """Plot comprehensive k-mer performance comparison."""
    n_levels = len(taxonomic_levels)
    fig, axes = plt.subplots(n_levels, 2, figsize=(15, 5*n_levels))
    
    level_names = {
        'd': 'Domain',
        'p': 'Phylum',
        'c': 'Class',
        'o': 'Order',
        'f': 'Family',
        'g': 'Genus',
        's': 'Species'
    }
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(results.keys())))
    
    for i, level in enumerate(taxonomic_levels):
        acc_ax = axes[i, 0]
        loss_ax = axes[i, 1]
        
        for (k, k_results), color in zip(results.items(), colors):
            if level in k_results and k_results[level] is not None:
                history = k_results[level]['history'].history
                
                acc_ax.plot(history['accuracy'], 
                          linestyle='-', 
                          color=color, 
                          label=f'k={k} (train)',
                          alpha=0.7)
                acc_ax.plot(history['val_accuracy'], 
                          linestyle='--', 
                          color=color, 
                          label=f'k={k} (val)',
                          alpha=0.7)
                
                loss_ax.plot(history['loss'], 
                           linestyle='-', 
                           color=color, 
                           label=f'k={k} (train)',
                           alpha=0.7)
                loss_ax.plot(history['val_loss'], 
                           linestyle='--', 
                           color=color, 
                           label=f'k={k} (val)',
                           alpha=0.7)
        
        acc_ax.set_title(f'{level_names.get(level, level)} Level - Accuracy vs Epochs')
        acc_ax.set_xlabel('Epochs')
        acc_ax.set_ylabel('Accuracy')
        acc_ax.grid(True, alpha=0.3)
        acc_ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        loss_ax.set_title(f'{level_names.get(level, level)} Level - Loss vs Epochs')
        loss_ax.set_xlabel('Epochs')
        loss_ax.set_ylabel('Loss')
        loss_ax.grid(True, alpha=0.3)
        loss_ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
    # Summary bar plot
    plt.figure(figsize=(12, 6))
    k_values = list(results.keys())
    x = np.arange(len(taxonomic_levels))
    width = 0.8 / len(k_values)
    
    for i, k in enumerate(k_values):
        accuracies = []
        for level in taxonomic_levels:
            if level in results[k] and results[k][level] is not None:
                accuracies.append(results[k][level]['test_accuracy'])
            else:
                accuracies.append(0)
        
        plt.bar(x + i*width - width*len(k_values)/2, 
               accuracies, 
               width, 
               label=f'k={k}',
               alpha=0.7)
    
    plt.xlabel('Taxonomic Level')
    plt.ylabel('Test Accuracy')
    plt.title('K-mer Performance Comparison Across Taxonomic Levels')
    plt.xticks(x, [level_names.get(level, level) for level in taxonomic_levels])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def optimize_k_value(df, k_range=range(5,7), taxonomic_levels=['d']):
    """Test different k-mer lengths and stop when high accuracy is achieved."""
    results = {}
    best_accuracies = {level: 0.0 for level in taxonomic_levels}
    accuracy_threshold = 0.95  # 95% accuracy threshold
    improvement_threshold = 0.005  # 0.5% improvement threshold
    
    for k in k_range:
        print(f"\n{'='*50}")
        print(f"Testing k={k}")
        print(f"{'='*50}")
        
        df['kmers'] = df['Feature ID'].apply(lambda x: create_kmers(x, k))
        
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(df['kmers']).toarray()
        
        level_results = {}
        improvements = []
        
        for level in taxonomic_levels:
            df[level] = df['Taxon'].apply(
                lambda x: x.split(f'{level}__')[1].split('; ')[0] if f'{level}__' in x else 'Unknown'
            )
            
            le = LabelEncoder()
            y = le.fit_transform(df[level])
            
            result = train_taxonomy_model(X, y, level, le, k)
            level_results[level] = result
            
            if result:
                current_accuracy = result['test_accuracy']
                improvement = current_accuracy - best_accuracies[level]
                best_accuracies[level] = max(best_accuracies[level], current_accuracy)
                improvements.append(improvement)
                
                print(f"\nLevel {level}:")
                print(f"Current accuracy: {current_accuracy:.4f}")
                print(f"Improvement: {improvement:.4f}")
        
        results[k] = level_results
        
        # Check stopping criteria
        avg_accuracy = np.mean([
            results[k][level]['test_accuracy'] 
            for level in taxonomic_levels 
            if level in results[k] and results[k][level] is not None
        ])
        
        avg_improvement = np.mean(improvements)
        
        print(f"\nAverage accuracy for k={k}: {avg_accuracy:.4f}")
        print(f"Average improvement: {avg_improvement:.4f}")
        
        if avg_accuracy >= accuracy_threshold:
            print(f"\nStopping: Achieved high accuracy threshold ({accuracy_threshold:.2f}) with k={k}")
            break
            
        if k > k_range[0] and avg_improvement < improvement_threshold:
            print(f"\nStopping: Minimal improvement ({avg_improvement:.4f} < {improvement_threshold:.4f}) with k={k}")
            break
    
    plot_kmer_comparison(results, taxonomic_levels)
    
    # Print final summary of best k value
    print("\nBest k-mer size summary:")
    for level in taxonomic_levels:
        best_k = max(
            results.keys(),
            key=lambda k: results[k][level]['test_accuracy'] if level in results[k] and results[k][level] is not None else -1
        )
        if level in results[best_k] and results[best_k][level] is not None:
            print(f"{level} Level - Best k={best_k} (Accuracy: {results[best_k][level]['test_accuracy']:.4f})")
    
    # Save final summary
    summary_path = "kmer_optimization_summary.json"
    summary = {}
    for k, level_results in results.items():
        summary[f"k{k}"] = {}
        for level, result in level_results.items():
            if result:
                summary[f"k{k}"][level] = {
                    "accuracy": float(result['test_accuracy']),
                    "loss": float(result['test_loss']),
                    "features": int(result['features']),
                    "classes": int(result['classes']),
                    "improvement": float(result['test_accuracy'] - best_accuracies.get(level, 0.0))
                }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\nDetailed k-mer optimization summary saved to {summary_path}")
    
    return results

# Main execution
if __name__ == "__main__":
    # Configure CPU
    configure_cpu()
    
    # Load data
    df = load_data()
    
    # Run optimization with early stopping
    k_results = optimize_k_value(df)
    
    # Print final results summary
    print("\nFinal Results Summary:")
    for k, level_results in k_results.items():
        print(f"\nK={k}")
        for level, result in level_results.items():
            if result:
                print(f"{level} Level - Accuracy: {result['test_accuracy']:.4f}, Loss: {result['test_loss']:.4f}")
    
    # Save final summary
    summary_path = "final_results_summary.json"
    summary = {}
    for k, level_results in k_results.items():
        summary[f"k{k}"] = {}
        for level, result in level_results.items():
            if result:
                summary[f"k{k}"][level] = {
                    "accuracy": float(result['test_accuracy']),
                    "loss": float(result['test_loss']),
                    "features": int(result['features']),
                    "classes": int(result['classes'])
                }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\nFinal summary saved to {summary_path}")
    
    # Plot final comparison
    plt.figure(figsize=(15, 8))
    for k in k_results.keys():
        accuracies = []
        levels = []
        for level, result in k_results[k].items():
            if result:
                accuracies.append(result['test_accuracy'])
                levels.append(level)
        plt.plot(levels, accuracies, marker='o', label=f'k={k}')
    
    plt.title('Performance Across Taxonomic Levels')
    plt.xlabel('Taxonomic Level')
    plt.ylabel('Test Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('final_comparison.png')
    plt.show()