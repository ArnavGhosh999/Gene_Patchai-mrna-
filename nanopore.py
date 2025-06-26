#!/usr/bin/env python3
"""
Simple Nanopore Sequencing Analysis Pipeline
Uses ont-fast5-api and h5py for FAST5 file processing
Independent of MinKNOW, with basic basecalling simulation
"""

import os
import json
import h5py
import numpy as np
from ont_fast5_api.fast5_interface import get_fast5_file
from ont_fast5_api.fast5_read import Fast5Read
from pathlib import Path
import argparse
from datetime import datetime

class NanoporeProcessor:
    def __init__(self, reference_fasta, dataset_json, assembly_jsonl):
        """Initialize the nanopore processor with reference files"""
        self.reference_fasta = reference_fasta
        self.dataset_json = dataset_json
        self.assembly_jsonl = assembly_jsonl
        self.reference_info = self.load_reference_info()
        
    def load_reference_info(self):
        """Load reference genome information from JSON files"""
        ref_info = {}
        
        # Load dataset catalog
        try:
            with open(self.dataset_json, 'r') as f:
                dataset_data = json.load(f)
                ref_info['dataset'] = dataset_data
                print(f"Loaded dataset info: {dataset_data.get('organism', 'Unknown')}")
        except Exception as e:
            print(f"Warning: Could not load dataset JSON: {e}")
            
        # Load assembly report
        try:
            with open(self.assembly_jsonl, 'r') as f:
                assembly_data = json.loads(f.readline())
                ref_info['assembly'] = assembly_data
                print(f"Assembly: {assembly_data.get('assemblyInfo', {}).get('assemblyName', 'Unknown')}")
        except Exception as e:
            print(f"Warning: Could not load assembly JSONL: {e}")
            
        return ref_info
    
    def read_fast5_file(self, fast5_path):
        """Read and extract data from FAST5 file using ont-fast5-api"""
        reads_data = []
        
        try:
            with get_fast5_file(fast5_path, mode="r") as f5:
                read_ids = f5.get_read_ids()
                print(f"Found {len(read_ids)} reads in {fast5_path}")
                
                for read_id in read_ids:
                    read = f5.get_read(read_id)
                    
                    # Get raw signal data
                    raw_data = read.get_raw_data()
                    
                    # Get channel info
                    channel_info = read.get_channel_info()
                    
                    # Basic read info
                    read_info = {
                        'read_id': read_id,
                        'channel': channel_info.get('channel_number', 0),
                        'start_time': channel_info.get('start_time', 0),
                        'duration': len(raw_data),
                        'raw_signal': raw_data,
                        'mean_signal': np.mean(raw_data),
                        'std_signal': np.std(raw_data)
                    }
                    
                    reads_data.append(read_info)
                    
        except Exception as e:
            print(f"Error reading FAST5 file {fast5_path}: {e}")
            # Fallback to direct h5py access
            reads_data = self.read_fast5_h5py(fast5_path)
            
        return reads_data
    
    def read_fast5_h5py(self, fast5_path):
        """Fallback method using h5py directly"""
        reads_data = []
        
        try:
            with h5py.File(fast5_path, 'r') as f5:
                # Navigate HDF5 structure
                for read_key in f5.keys():
                    if read_key.startswith('read_'):
                        read_group = f5[read_key]
                        
                        # Get channel info
                        channel_number = 1
                        start_time = 0
                        
                        if 'channel_id' in read_group:
                            channel_attrs = read_group['channel_id'].attrs
                            channel_number = channel_attrs.get('channel_number', 1)
                        elif 'UniqueGlobalKey' in read_group:
                            unique_attrs = read_group['UniqueGlobalKey'].attrs
                            channel_number = unique_attrs.get('channel_id', 1)
                            start_time = unique_attrs.get('start_time', 0)
                        
                        # Get raw signal
                        if 'Raw' in read_group and 'Signal' in read_group['Raw']:
                            signal = read_group['Raw']['Signal'][:]
                            raw_attrs = read_group['Raw'].attrs
                            start_time = raw_attrs.get('start_time', start_time)
                            
                            read_info = {
                                'read_id': read_key,
                                'channel': channel_number,
                                'start_time': start_time,
                                'duration': len(signal),
                                'raw_signal': signal,
                                'mean_signal': np.mean(signal),
                                'std_signal': np.std(signal)
                            }
                            reads_data.append(read_info)
                            
        except Exception as e:
            print(f"Error with h5py fallback: {e}")
            
        return reads_data
    
    def simple_basecall(self, raw_signal):
        """
        Simple basecalling simulation (replace with actual Guppy integration)
        This is a placeholder - in real use, integrate with Guppy basecaller
        """
        # Normalize signal
        normalized = (raw_signal - np.mean(raw_signal)) / np.std(raw_signal)
        
        # Simple threshold-based calling (very basic)
        bases = []
        window_size = 10
        
        for i in range(0, len(normalized) - window_size, window_size):
            window = normalized[i:i + window_size]
            mean_val = np.mean(window)
            
            # Simple mapping based on signal levels
            if mean_val > 1.0:
                bases.append('A')
            elif mean_val > 0.0:
                bases.append('T')
            elif mean_val > -1.0:
                bases.append('G')
            else:
                bases.append('C')
                
        return ''.join(bases)
    
    def process_reads(self, reads_data):
        """Process reads and perform basecalling"""
        processed_reads = []
        
        for read_info in reads_data:
            # Perform basecalling
            sequence = self.simple_basecall(read_info['raw_signal'])
            
            processed_read = {
                'read_id': read_info['read_id'],
                'channel': read_info['channel'],
                'sequence': sequence,
                'quality_score': self.calculate_quality_score(read_info),
                'length': len(sequence),
                'mean_signal': read_info['mean_signal'],
                'duration': read_info['duration']
            }
            
            processed_reads.append(processed_read)
            
        return processed_reads
    
    def calculate_quality_score(self, read_info):
        """Calculate a simple quality score based on signal characteristics"""
        # Simple quality metric based on signal stability
        signal_ratio = read_info['std_signal'] / (read_info['mean_signal'] + 1e-6)
        quality = max(0, min(40, 40 - signal_ratio * 10))
        return quality
    
    def save_results(self, processed_reads, output_dir):
        """Save processed results to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as FASTQ
        fastq_path = os.path.join(output_dir, 'basecalled_reads.fastq')
        with open(fastq_path, 'w') as f:
            for read in processed_reads:
                # FASTQ format
                f.write(f"@{read['read_id']}\n")
                f.write(f"{read['sequence']}\n")
                f.write("+\n")
                f.write('I' * len(read['sequence']) + '\n')  # Placeholder quality
        
        # Save summary statistics
        stats_path = os.path.join(output_dir, 'run_statistics.json')
        stats = {
            'total_reads': len(processed_reads),
            'total_bases': sum(r['length'] for r in processed_reads),
            'mean_read_length': np.mean([r['length'] for r in processed_reads]),
            'mean_quality': np.mean([r['quality_score'] for r in processed_reads]),
            'processing_time': datetime.now().isoformat(),
            'reference_genome': os.path.basename(self.reference_fasta)
        }
        
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Results saved to {output_dir}")
        print(f"Total reads processed: {stats['total_reads']}")
        print(f"Total bases called: {stats['total_bases']}")
        
        return fastq_path, stats_path

def create_sample_fast5(output_path, num_reads=5):
    """Create a sample FAST5 file for testing with proper ont-fast5-api structure"""
    try:
        with h5py.File(output_path, 'w') as f5:
            for i in range(num_reads):
                read_id = f"read_{i:04d}"
                read_group = f5.create_group(read_id)
                
                # Create channel_id group (required by ont-fast5-api)
                channel_group = read_group.create_group('channel_id')
                channel_group.attrs['channel_number'] = i % 4 + 1
                channel_group.attrs['digitisation'] = 8192.0
                channel_group.attrs['offset'] = 10.0
                channel_group.attrs['range'] = 1400.0
                channel_group.attrs['sampling_rate'] = 4000.0
                
                # Create Raw data group
                raw_group = read_group.create_group('Raw')
                
                # Generate synthetic signal data
                signal_length = np.random.randint(1000, 5000)
                synthetic_signal = np.random.normal(100, 20, signal_length).astype(np.int16)
                
                # Add some structured patterns to make it more realistic
                for j in range(0, len(synthetic_signal), 50):
                    if j + 10 < len(synthetic_signal):
                        synthetic_signal[j:j+10] += np.random.choice([50, -50, 0], 10)
                
                raw_group.create_dataset('Signal', data=synthetic_signal)
                raw_group.attrs['start_time'] = i * 1000
                raw_group.attrs['read_id'] = read_id
                raw_group.attrs['read_number'] = i
                
                # Add UniqueGlobalKey group (required)
                unique_group = read_group.create_group('UniqueGlobalKey')
                unique_group.attrs['channel_id'] = i % 4 + 1
                unique_group.attrs['start_time'] = i * 1000
                
        print(f"Created sample FAST5 file: {output_path}")
        return True
    except Exception as e:
        print(f"Error creating sample FAST5: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Simple Nanopore Processing Pipeline')
    parser.add_argument('--fast5_dir', help='Directory containing FAST5 files')
    parser.add_argument('--create_sample', action='store_true', help='Create sample FAST5 files for testing')
    parser.add_argument('--reference', default=r'ncbi_dataset\ncbi_dataset\data\GCA_000001405.29_GRCh38.p14_genomic.fna',
                       help='Reference FASTA file')
    parser.add_argument('--dataset_json', default=r'ncbi_dataset\ncbi_dataset\data\dataset_catalog.json',
                       help='Dataset catalog JSON')
    parser.add_argument('--assembly_jsonl', default=r'ncbi_dataset\ncbi_dataset\data\assembly_data_report.jsonl',
                       help='Assembly report JSONL')
    parser.add_argument('--output_dir', default='nanopore_output', help='Output directory')
    
    args = parser.parse_args()
    
    # Create sample data if requested
    if args.create_sample:
        sample_dir = 'sample_fast5'
        os.makedirs(sample_dir, exist_ok=True)
        
        # Create a few sample FAST5 files
        for i in range(3):
            sample_file = os.path.join(sample_dir, f'sample_{i}.fast5')
            create_sample_fast5(sample_file, num_reads=5)
        
        print(f"Sample FAST5 files created in {sample_dir}")
        args.fast5_dir = sample_dir
    
    # Check if fast5_dir is provided
    if not args.fast5_dir:
        print("Error: Please provide --fast5_dir or use --create_sample to generate test data")
        print("\nExamples:")
        print("  python nanopore.py --create_sample")
        print("  python nanopore.py --fast5_dir path/to/your/fast5/files")
        return
    
    # Initialize processor
    processor = NanoporeProcessor(
        reference_fasta=args.reference,
        dataset_json=args.dataset_json,
        assembly_jsonl=args.assembly_jsonl
    )
    
    # Process all FAST5 files in directory
    fast5_files = list(Path(args.fast5_dir).glob('*.fast5'))
    
    if not fast5_files:
        print(f"No FAST5 files found in {args.fast5_dir}")
        if not args.create_sample:
            print("Try using --create_sample to generate test data")
        return
    
    print(f"Processing {len(fast5_files)} FAST5 files...")
    
    all_processed_reads = []
    
    for fast5_file in fast5_files:
        print(f"Processing {fast5_file}...")
        
        # Read FAST5 data
        reads_data = processor.read_fast5_file(str(fast5_file))
        
        if reads_data:
            # Process reads (basecalling)
            processed_reads = processor.process_reads(reads_data)
            all_processed_reads.extend(processed_reads)
    
    if all_processed_reads:
        # Save results
        processor.save_results(all_processed_reads, args.output_dir)
    else:
        print("No reads were successfully processed")

if __name__ == "__main__":
    main()

# Example usage:
# Create sample data and run:
# python nanopore.py --create_sample

# Or with existing FAST5 files:
# python nanopore.py --fast5_dir /path/to/fast5_files --output_dir results

# For batch processing with integration:
"""
# Install requirements:
pip install ont-fast5-api h5py numpy

# Run the pipeline with your Windows paths:
python nanopore.py --create_sample

# Or with existing FAST5 files:
python nanopore.py --fast5_dir C:\\path\\to\\your\\fast5_files
"""