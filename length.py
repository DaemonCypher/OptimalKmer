import pandas as pd
import numpy as np

def analyze_sequence_lengths(file_path="species_taxonomy_data.tsv"):
    """
    Analyze sequence lengths in the Feature ID column of the taxonomy dataset.
    
    Args:
        file_path (str): Path to the TSV file containing taxonomy data
        
    Returns:
        tuple: (shortest_seq, longest_seq, length_stats)
    """
    try:
        # Read the TSV file
        print("Loading dataset...")
        df = pd.read_csv(file_path, sep='\t')
        
        # Get sequences from Feature ID column
        sequences = df['Feature ID'].values
        
        # Calculate lengths of all sequences
        sequence_lengths = np.array([len(seq) for seq in sequences])
        
        # Find shortest sequence
        min_length = sequence_lengths.min()
        shortest_seq = sequences[sequence_lengths.argmin()]
        
        # Find longest sequence
        max_length = sequence_lengths.max()
        longest_seq = sequences[sequence_lengths.argmax()]
        
        # Calculate additional statistics
        length_stats = {
            'min_length': min_length,
            'max_length': max_length,
            'mean_length': sequence_lengths.mean(),
            'median_length': np.median(sequence_lengths),
            'std_length': sequence_lengths.std(),
            'total_sequences': len(sequences)
        }
        
        # Print results
        print("\nSequence Length Analysis Results:")
        print("-" * 50)
        print(f"Total number of sequences: {length_stats['total_sequences']:,}")
        print(f"\nLength Statistics:")
        print(f"Minimum length: {length_stats['min_length']:,}")
        print(f"Maximum length: {length_stats['max_length']:,}")
        print(f"Mean length: {length_stats['mean_length']:.2f}")
        print(f"Median length: {length_stats['median_length']:.2f}")
        print(f"Standard deviation: {length_stats['std_length']:.2f}")
        
        print("\nShortest Sequence:")
        print(f"Length: {min_length}")
        print(f"Sequence: {shortest_seq}")
        
        print("\nLongest Sequence:")
        print(f"Length: {max_length}")
        print(f"Sequence: {longest_seq}")
        
        return shortest_seq, longest_seq, length_stats
        
    except FileNotFoundError:
        print(f"Error: Could not find file '{file_path}'")
    except Exception as e:
        print(f"Error analyzing sequences: {str(e)}")

if __name__ == "__main__":
    analyze_sequence_lengths("species_taxonomy_data.tsv")