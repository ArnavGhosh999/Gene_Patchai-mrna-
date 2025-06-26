#!/usr/bin/env python3
"""
FastQC-style Quality Control Pipeline
Uses fastqc-py for comprehensive sequence quality assessment
Independent QC pipeline with MultiQC-style batch reporting
Performs QC checks on raw sequence data before alignment/assembly
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from datetime import datetime
from collections import defaultdict, Counter
import gzip
import re
import warnings
warnings.filterwarnings('ignore')

class FastQCProcessor:
    def __init__(self, output_dir='qc_output'):
        """Initialize FastQC processor"""
        self.output_dir = output_dir
        self.samples_data = {}
        self.batch_summary = {}
        
        # Common adapter sequences
        self.adapters = {
            'Illumina_Universal': 'AGATCGGAAGAG',
            'Illumina_Small_RNA': 'TGGAATTCTCGG',
            'Nextera': 'CTGTCTCTTATA',
            'TruSeq_R1': 'AGATCGGAAGAGCACACGTCTGAACTCCAGTCA',
            'TruSeq_R2': 'AGATCGGAAGAGCGTCGTGTAGGGAAAGAGTGT',
            'Nanopore_Native': 'AATGTACTTCGTTCAGTTACGTATTGCT'
        }
        
        os.makedirs(output_dir, exist_ok=True)
    
    def create_sample_fastq(self, output_path, num_reads=10000, read_length=150):
        """Create sample FASTQ file for testing"""
        try:
            with open(output_path, 'w') as fq:
                for i in range(num_reads):
                    # Generate realistic sequence with quality variation
                    read_id = f"@read_{i:06d}"
                    
                    # Generate sequence with some patterns
                    bases = ['A', 'T', 'G', 'C']
                    sequence = ''
                    
                    # Add some realistic patterns
                    if i % 100 == 0:  # 1% adapter contamination
                        adapter_seq = self.adapters['Illumina_Universal']
                        remaining_length = read_length - len(adapter_seq)
                        if remaining_length > 0:
                            sequence = adapter_seq + ''.join(np.random.choice(bases, size=remaining_length))
                        else:
                            sequence = adapter_seq[:read_length]
                    elif i % 50 == 0:  # 2% low complexity
                        poly_a_length = min(20, read_length)
                        remaining_length = read_length - poly_a_length
                        if remaining_length > 0:
                            sequence = 'A' * poly_a_length + ''.join(np.random.choice(bases, size=remaining_length))
                        else:
                            sequence = 'A' * read_length
                    else:  # Normal sequence
                        # Simulate GC bias (human genome ~41% GC)
                        sequence = ''
                        for pos in range(read_length):
                            if np.random.random() < 0.41:
                                sequence += np.random.choice(['G', 'C'])
                            else:
                                sequence += np.random.choice(['A', 'T'])
                    
                    # Ensure sequence is exactly read_length
                    if len(sequence) > read_length:
                        sequence = sequence[:read_length]
                    elif len(sequence) < read_length:
                        sequence += ''.join(np.random.choice(bases, size=read_length - len(sequence)))
                    
                    # Generate quality scores (Phred+33)
                    # Simulate quality drop towards 3' end
                    qualities = []
                    for pos in range(read_length):
                        if pos < int(read_length * 0.8):  # First 80% high quality
                            qual = np.random.randint(30, 40)
                        else:  # Last 20% declining quality
                            decay_factor = (pos - int(read_length * 0.8))
                            qual = max(10, np.random.randint(15, 35) - decay_factor)
                        qualities.append(chr(int(qual) + 33))
                    
                    quality_string = ''.join(qualities)
                    
                    # Write FASTQ record
                    fq.write(f"{read_id}\n")
                    fq.write(f"{sequence}\n")
                    fq.write("+\n")
                    fq.write(f"{quality_string}\n")
            
            print(f"Created sample FASTQ file: {output_path}")
            return True
        except Exception as e:
            print(f"Error creating sample FASTQ: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_sample_file(self, fastq_path):
        """Test and debug sample file creation"""
        if not os.path.exists(fastq_path):
            print(f"File does not exist: {fastq_path}")
            return False
        
        print(f"File exists: {fastq_path}")
        print(f"File size: {os.path.getsize(fastq_path)} bytes")
        
        # Read first few lines to check format
        try:
            with open(fastq_path, 'r') as f:
                lines = [f.readline().strip() for _ in range(8)]
            
            print("First 8 lines:")
            for i, line in enumerate(lines):
                print(f"  {i}: {repr(line)}")
            
            return True
        except Exception as e:
            print(f"Error reading file: {e}")
            return False
    
    def read_fastq_file(self, fastq_path):
        """Read FASTQ file and extract sequence data"""
        sequences = []
        qualities = []
        sequence_lengths = []
        
        try:
            # Handle both gzipped and uncompressed files
            if fastq_path.endswith('.gz'):
                file_handle = gzip.open(fastq_path, 'rt')
            else:
                file_handle = open(fastq_path, 'r')
            
            with file_handle as fq:
                lines = []
                for line in fq:
                    lines.append(line.strip())
                
                # Process lines in groups of 4 (FASTQ format)
                for i in range(0, len(lines), 4):
                    if i + 3 < len(lines):
                        header = lines[i]
                        sequence = lines[i + 1]
                        plus = lines[i + 2]
                        quality = lines[i + 3]
                        
                        # Validate FASTQ format
                        if header.startswith('@') and plus.startswith('+') and sequence and quality:
                            if len(sequence) == len(quality):  # Quality string should match sequence length
                                sequences.append(sequence)
                                qualities.append(quality)
                                sequence_lengths.append(len(sequence))
                
                print(f"Read {len(sequences)} sequences from {fastq_path}")
                
                # Debug: print first few sequences if none found
                if len(sequences) == 0 and len(lines) > 0:
                    print(f"Debug: First 10 lines from {fastq_path}:")
                    for i, line in enumerate(lines[:10]):
                        print(f"  Line {i}: {repr(line)}")
                
                return sequences, qualities, sequence_lengths
            
        except Exception as e:
            print(f"Error reading FASTQ file {fastq_path}: {e}")
            return [], [], []
    
    def calculate_basic_stats(self, sequences, qualities):
        """Calculate basic sequence statistics"""
        if not sequences:
            return {}
        
        total_sequences = len(sequences)
        sequence_lengths = [len(seq) for seq in sequences]
        total_bases = sum(sequence_lengths)
        
        stats = {
            'total_sequences': total_sequences,
            'total_bases': total_bases,
            'sequence_length_min': min(sequence_lengths),
            'sequence_length_max': max(sequence_lengths),
            'sequence_length_mean': np.mean(sequence_lengths),
            'sequence_length_median': np.median(sequence_lengths)
        }
        
        return stats
    
    def analyze_base_quality(self, qualities):
        """Analyze per-base quality scores"""
        if not qualities:
            return {}
        
        # Convert quality strings to numeric scores
        quality_scores = []
        position_qualities = defaultdict(list)
        
        for qual_string in qualities:
            qual_nums = [ord(q) - 33 for q in qual_string]
            quality_scores.extend(qual_nums)
            
            for pos, qual in enumerate(qual_nums):
                position_qualities[pos].append(qual)
        
        # Calculate per-position statistics
        per_position_stats = {}
        for pos in sorted(position_qualities.keys()):
            quals = position_qualities[pos]
            per_position_stats[pos] = {
                'mean': np.mean(quals),
                'median': np.median(quals),
                'q25': np.percentile(quals, 25),
                'q75': np.percentile(quals, 75),
                'min': min(quals),
                'max': max(quals)
            }
        
        quality_analysis = {
            'mean_quality_score': np.mean(quality_scores),
            'median_quality_score': np.median(quality_scores),
            'q20_bases_percent': (np.sum(np.array(quality_scores) >= 20) / len(quality_scores)) * 100,
            'q30_bases_percent': (np.sum(np.array(quality_scores) >= 30) / len(quality_scores)) * 100,
            'per_position_quality': per_position_stats
        }
        
        return quality_analysis
    
    def analyze_gc_content(self, sequences):
        """Analyze GC content distribution"""
        if not sequences:
            return {}
        
        gc_contents = []
        total_bases = {'A': 0, 'T': 0, 'G': 0, 'C': 0, 'N': 0}
        
        for sequence in sequences:
            gc_count = sequence.count('G') + sequence.count('C')
            gc_content = (gc_count / len(sequence)) * 100 if len(sequence) > 0 else 0
            gc_contents.append(gc_content)
            
            # Count individual bases
            for base in sequence:
                if base in total_bases:
                    total_bases[base] += 1
                else:
                    total_bases['N'] += 1
        
        total_counted = sum(total_bases.values())
        base_percentages = {base: (count / total_counted) * 100 
                          for base, count in total_bases.items()}
        
        gc_analysis = {
            'mean_gc_content': np.mean(gc_contents),
            'median_gc_content': np.median(gc_contents),
            'gc_content_distribution': gc_contents,
            'base_composition': base_percentages,
            'theoretical_gc': 50.0  # For comparison
        }
        
        return gc_analysis
    
    def detect_adapter_contamination(self, sequences):
        """Detect adapter contamination"""
        if not sequences:
            return {}
        
        adapter_hits = defaultdict(int)
        contaminated_sequences = 0
        adapter_positions = defaultdict(list)
        
        for seq_idx, sequence in enumerate(sequences):
            seq_contaminated = False
            
            for adapter_name, adapter_seq in self.adapters.items():
                # Look for adapter at beginning and end of reads
                for position in [0, len(sequence) - len(adapter_seq)]:
                    if position >= 0:
                        region = sequence[position:position + len(adapter_seq)]
                        
                        # Allow some mismatches
                        matches = sum(1 for a, b in zip(adapter_seq, region) if a == b)
                        match_rate = matches / len(adapter_seq) if len(adapter_seq) > 0 else 0
                        
                        if match_rate >= 0.8:  # 80% match threshold
                            adapter_hits[adapter_name] += 1
                            adapter_positions[adapter_name].append(position)
                            seq_contaminated = True
            
            if seq_contaminated:
                contaminated_sequences += 1
        
        contamination_rate = (contaminated_sequences / len(sequences)) * 100
        
        adapter_analysis = {
            'total_contaminated_sequences': contaminated_sequences,
            'contamination_rate_percent': contamination_rate,
            'adapter_hits': dict(adapter_hits),
            'adapter_positions': {k: v for k, v in adapter_positions.items()},
            'most_common_adapter': max(adapter_hits.items(), key=lambda x: x[1])[0] if adapter_hits else None
        }
        
        return adapter_analysis
    
    def detect_overrepresented_sequences(self, sequences, min_count=10):
        """Detect overrepresented sequences"""
        if not sequences:
            return {}
        
        # Count k-mers of different lengths
        kmer_counts = defaultdict(Counter)
        
        for seq in sequences:
            # Check different k-mer sizes
            for k in [10, 15, 20]:
                if len(seq) >= k:
                    for i in range(len(seq) - k + 1):
                        kmer = seq[i:i+k]
                        kmer_counts[k][kmer] += 1
        
        # Find overrepresented sequences
        overrepresented = {}
        total_sequences = len(sequences)
        
        for k, counter in kmer_counts.items():
            overrep_kmers = []
            for kmer, count in counter.most_common(20):  # Top 20
                frequency = (count / total_sequences) * 100
                if count >= min_count and frequency > 0.1:  # >0.1% frequency
                    overrep_kmers.append({
                        'sequence': kmer,
                        'count': count,
                        'frequency_percent': frequency,
                        'possible_source': self.identify_sequence_source(kmer)
                    })
            
            if overrep_kmers:
                overrepresented[f'{k}mer'] = overrep_kmers
        
        return overrepresented
    
    def identify_sequence_source(self, sequence):
        """Identify possible source of overrepresented sequence"""
        # Check against known adapters
        for adapter_name, adapter_seq in self.adapters.items():
            if sequence in adapter_seq or adapter_seq in sequence:
                return f"Adapter: {adapter_name}"
        
        # Check for low complexity
        if len(set(sequence)) <= 2:
            return "Low complexity"
        
        # Check for repetitive patterns
        if len(sequence) >= 6:
            for i in range(2, len(sequence)//2 + 1):
                pattern = sequence[:i]
                if sequence.startswith(pattern * (len(sequence)//i)):
                    return f"Repetitive pattern: {pattern}"
        
        return "Unknown"
    
    def analyze_sequence_duplication(self, sequences):
        """Analyze sequence duplication levels"""
        if not sequences:
            return {}
        
        sequence_counts = Counter(sequences)
        total_sequences = len(sequences)
        unique_sequences = len(sequence_counts)
        
        # Count duplication levels
        duplication_levels = Counter()
        for seq, count in sequence_counts.items():
            duplication_levels[count] += 1
        
        duplication_analysis = {
            'total_sequences': total_sequences,
            'unique_sequences': unique_sequences,
            'duplication_rate_percent': ((total_sequences - unique_sequences) / total_sequences) * 100,
            'duplication_levels': dict(duplication_levels),
            'most_duplicated_count': max(sequence_counts.values()),
            'sequences_appearing_once': duplication_levels[1]
        }
        
        return duplication_analysis
    
    def generate_qc_plots(self, sample_name, quality_analysis, gc_analysis):
        """Generate QC plots for a sample"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Quality Control Report: {sample_name}', fontsize=16)
        
        # Plot 1: Per-position quality
        if quality_analysis.get('per_position_quality'):
            positions = sorted(quality_analysis['per_position_quality'].keys())
            means = [quality_analysis['per_position_quality'][pos]['mean'] for pos in positions]
            q25s = [quality_analysis['per_position_quality'][pos]['q25'] for pos in positions]
            q75s = [quality_analysis['per_position_quality'][pos]['q75'] for pos in positions]
            
            axes[0, 0].plot(positions, means, 'b-', label='Mean')
            axes[0, 0].fill_between(positions, q25s, q75s, alpha=0.3, color='blue')
            axes[0, 0].axhline(y=20, color='orange', linestyle='--', label='Q20')
            axes[0, 0].axhline(y=30, color='red', linestyle='--', label='Q30')
            axes[0, 0].set_xlabel('Position in read')
            axes[0, 0].set_ylabel('Quality score')
            axes[0, 0].set_title('Per-position Quality Scores')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: GC content distribution
        if gc_analysis.get('gc_content_distribution'):
            axes[0, 1].hist(gc_analysis['gc_content_distribution'], bins=50, alpha=0.7, color='green')
            axes[0, 1].axvline(gc_analysis['mean_gc_content'], color='red', linestyle='--', 
                              label=f"Mean: {gc_analysis['mean_gc_content']:.1f}%")
            axes[0, 1].axvline(50, color='orange', linestyle='--', label='Theoretical: 50%')
            axes[0, 1].set_xlabel('GC Content (%)')
            axes[0, 1].set_ylabel('Number of reads')
            axes[0, 1].set_title('GC Content Distribution')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Base composition
        if gc_analysis.get('base_composition'):
            bases = ['A', 'T', 'G', 'C']
            percentages = [gc_analysis['base_composition'].get(base, 0) for base in bases]
            colors = ['red', 'blue', 'green', 'orange']
            
            axes[1, 0].bar(bases, percentages, color=colors, alpha=0.7)
            axes[1, 0].set_ylabel('Percentage')
            axes[1, 0].set_title('Base Composition')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Add percentage labels on bars
            for i, (base, pct) in enumerate(zip(bases, percentages)):
                axes[1, 0].text(i, pct + 0.5, f'{pct:.1f}%', ha='center')
        
        # Plot 4: Quality score distribution
        if quality_analysis.get('q20_bases_percent'):
            categories = ['Q20+ Bases', 'Q30+ Bases', 'Below Q20']
            q20_pct = quality_analysis['q20_bases_percent']
            q30_pct = quality_analysis['q30_bases_percent']
            below_q20 = 100 - q20_pct
            
            values = [q20_pct, q30_pct, below_q20]
            colors = ['lightgreen', 'green', 'red']
            
            axes[1, 1].pie(values, labels=categories, colors=colors, autopct='%1.1f%%')
            axes[1, 1].set_title('Quality Distribution')
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, f'{sample_name}_qc_plots.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def process_sample(self, fastq_path):
        """Process a single FASTQ sample"""
        sample_name = Path(fastq_path).stem
        print(f"Processing sample: {sample_name}")
        
        # Read FASTQ data
        sequences, qualities, seq_lengths = self.read_fastq_file(fastq_path)
        
        if not sequences:
            print(f"No sequences found in {fastq_path}")
            return None
        
        # Perform all QC analyses
        basic_stats = self.calculate_basic_stats(sequences, qualities)
        quality_analysis = self.analyze_base_quality(qualities)
        gc_analysis = self.analyze_gc_content(sequences)
        adapter_analysis = self.detect_adapter_contamination(sequences)
        overrep_analysis = self.detect_overrepresented_sequences(sequences)
        duplication_analysis = self.analyze_sequence_duplication(sequences)
        
        # Generate plots
        plot_path = self.generate_qc_plots(sample_name, quality_analysis, gc_analysis)
        
        # Compile results
        sample_results = {
            'sample_name': sample_name,
            'file_path': fastq_path,
            'basic_statistics': basic_stats,
            'quality_analysis': quality_analysis,
            'gc_analysis': gc_analysis,
            'adapter_contamination': adapter_analysis,
            'overrepresented_sequences': overrep_analysis,
            'sequence_duplication': duplication_analysis,
            'qc_plots': plot_path,
            'timestamp': datetime.now().isoformat()
        }
        
        # Store in class data
        self.samples_data[sample_name] = sample_results
        
        # Save individual sample report
        sample_report_path = os.path.join(self.output_dir, f'{sample_name}_qc_report.json')
        with open(sample_report_path, 'w') as f:
            json.dump(sample_results, f, indent=2, default=str)
        
        print(f"Sample {sample_name} QC complete")
        return sample_results
    
    def generate_batch_report(self):
        """Generate MultiQC-style batch report"""
        if not self.samples_data:
            print("No samples processed for batch report")
            return None
        
        # Aggregate statistics across samples
        batch_stats = {
            'total_samples': len(self.samples_data),
            'processing_date': datetime.now().isoformat(),
            'samples_summary': {}
        }
        
        # Collect key metrics from all samples
        for sample_name, data in self.samples_data.items():
            sample_summary = {
                'total_sequences': data['basic_statistics']['total_sequences'],
                'total_bases': data['basic_statistics']['total_bases'],
                'mean_quality': data['quality_analysis']['mean_quality_score'],
                'q30_percent': data['quality_analysis']['q30_bases_percent'],
                'gc_content': data['gc_analysis']['mean_gc_content'],
                'adapter_contamination': data['adapter_contamination']['contamination_rate_percent'],
                'duplication_rate': data['sequence_duplication']['duplication_rate_percent']
            }
            batch_stats['samples_summary'][sample_name] = sample_summary
        
        # Generate batch summary plots
        self.create_batch_plots(batch_stats)
        
        # Save batch report
        batch_report_path = os.path.join(self.output_dir, 'batch_qc_report.json')
        with open(batch_report_path, 'w') as f:
            json.dump(batch_stats, f, indent=2, default=str)
        
        # Create HTML summary report
        html_report_path = self.create_html_report(batch_stats)
        
        print(f"Batch report saved: {batch_report_path}")
        print(f"HTML report saved: {html_report_path}")
        
        return batch_report_path, html_report_path
    
    def create_batch_plots(self, batch_stats):
        """Create batch comparison plots"""
        samples = list(batch_stats['samples_summary'].keys())
        
        if len(samples) < 2:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Batch Quality Control Summary', fontsize=16)
        
        # Plot 1: Quality scores comparison
        q30_values = [batch_stats['samples_summary'][s]['q30_percent'] for s in samples]
        mean_qual_values = [batch_stats['samples_summary'][s]['mean_quality'] for s in samples]
        
        axes[0, 0].bar(range(len(samples)), q30_values, alpha=0.7, color='green')
        axes[0, 0].set_xlabel('Samples')
        axes[0, 0].set_ylabel('Q30+ Bases (%)')
        axes[0, 0].set_title('Q30+ Percentage by Sample')
        axes[0, 0].set_xticks(range(len(samples)))
        axes[0, 0].set_xticklabels(samples, rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: GC content comparison
        gc_values = [batch_stats['samples_summary'][s]['gc_content'] for s in samples]
        axes[0, 1].bar(range(len(samples)), gc_values, alpha=0.7, color='blue')
        axes[0, 1].axhline(y=50, color='red', linestyle='--', label='Theoretical 50%')
        axes[0, 1].set_xlabel('Samples')
        axes[0, 1].set_ylabel('GC Content (%)')
        axes[0, 1].set_title('GC Content by Sample')
        axes[0, 1].set_xticks(range(len(samples)))
        axes[0, 1].set_xticklabels(samples, rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Adapter contamination
        adapter_values = [batch_stats['samples_summary'][s]['adapter_contamination'] for s in samples]
        axes[1, 0].bar(range(len(samples)), adapter_values, alpha=0.7, color='red')
        axes[1, 0].set_xlabel('Samples')
        axes[1, 0].set_ylabel('Adapter Contamination (%)')
        axes[1, 0].set_title('Adapter Contamination by Sample')
        axes[1, 0].set_xticks(range(len(samples)))
        axes[1, 0].set_xticklabels(samples, rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Read counts
        read_counts = [batch_stats['samples_summary'][s]['total_sequences'] for s in samples]
        axes[1, 1].bar(range(len(samples)), read_counts, alpha=0.7, color='purple')
        axes[1, 1].set_xlabel('Samples')
        axes[1, 1].set_ylabel('Total Reads')
        axes[1, 1].set_title('Read Count by Sample')
        axes[1, 1].set_xticks(range(len(samples)))
        axes[1, 1].set_xticklabels(samples, rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        batch_plot_path = os.path.join(self.output_dir, 'batch_comparison_plots.png')
        plt.savefig(batch_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return batch_plot_path
    
    def create_html_report(self, batch_stats):
        """Create HTML summary report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>FastQC Batch Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .pass {{ color: green; }}
                .warn {{ color: orange; }}
                .fail {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>FastQC Batch Quality Control Report</h1>
            <p>Generated: {batch_stats['processing_date']}</p>
            <p>Total Samples: {batch_stats['total_samples']}</p>
            
            <h2>Sample Summary</h2>
            <table>
                <tr>
                    <th>Sample</th>
                    <th>Total Reads</th>
                    <th>Mean Quality</th>
                    <th>Q30+ %</th>
                    <th>GC Content %</th>
                    <th>Adapter Contamination %</th>
                    <th>Duplication Rate %</th>
                </tr>
        """
        
        for sample, data in batch_stats['samples_summary'].items():
            # Determine status classes
            q30_class = 'pass' if data['q30_percent'] > 80 else 'warn' if data['q30_percent'] > 60 else 'fail'
            adapter_class = 'pass' if data['adapter_contamination'] < 5 else 'warn' if data['adapter_contamination'] < 10 else 'fail'
            
            html_content += f"""
                <tr>
                    <td>{sample}</td>
                    <td>{data['total_sequences']:,}</td>
                    <td>{data['mean_quality']:.1f}</td>
                    <td class="{q30_class}">{data['q30_percent']:.1f}</td>
                    <td>{data['gc_content']:.1f}</td>
                    <td class="{adapter_class}">{data['adapter_contamination']:.1f}</td>
                    <td>{data['duplication_rate']:.1f}</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <h2>Quality Thresholds</h2>
            <p><span class="pass">PASS:</span> Q30+ > 80%, Adapter contamination < 5%</p>
            <p><span class="warn">WARN:</span> Q30+ 60-80%, Adapter contamination 5-10%</p>
            <p><span class="fail">FAIL:</span> Q30+ < 60%, Adapter contamination > 10%</p>
        </body>
        </html>
        """
        
        html_path = os.path.join(self.output_dir, 'batch_qc_report.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return html_path

def main():
    parser = argparse.ArgumentParser(description='FastQC-style Quality Control Pipeline')
    parser.add_argument('--fastq_files', nargs='+', help='FASTQ files to analyze')
    parser.add_argument('--fastq_dir', help='Directory containing FASTQ files')
    parser.add_argument('--create_sample', action='store_true', help='Create sample FASTQ files for testing')
    parser.add_argument('--output_dir', default='qc_output', help='Output directory')
    parser.add_argument('--num_samples', type=int, default=3, help='Number of sample files to create')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = FastQCProcessor(output_dir=args.output_dir)
    
    # Get FASTQ files to process
    fastq_files = []
    
    if args.create_sample:
        print(f"Creating {args.num_samples} sample FASTQ files...")
        sample_dir = 'sample_fastq'
        os.makedirs(sample_dir, exist_ok=True)
        
        for i in range(args.num_samples):
            sample_file = os.path.join(sample_dir, f'sample_{i}.fastq')
            processor.create_sample_fastq(sample_file, num_reads=np.random.randint(5000, 15000))
            
            # Test the created file
            print(f"Testing created file: {sample_file}")
            processor.test_sample_file(sample_file)
            
            fastq_files.append(sample_file)
    
    if args.fastq_files:
        fastq_files.extend(args.fastq_files)
    
    if args.fastq_dir:
        fastq_dir = Path(args.fastq_dir)
        if fastq_dir.exists():
            # Find all FASTQ files in directory
            for pattern in ['*.fastq', '*.fq', '*.fastq.gz', '*.fq.gz']:
                fastq_files.extend(list(fastq_dir.glob(pattern)))
        else:
            print(f"Directory {args.fastq_dir} not found")
    
    if not fastq_files:
        print("No FASTQ files found. Use --create_sample to generate test data")
        print("Examples:")
        print("  python fastqc.py --create_sample")
        print("  python fastqc.py --fastq_files file1.fastq file2.fastq")
        print("  python fastqc.py --fastq_dir /path/to/fastq/files")
        return
    
    print(f"Processing {len(fastq_files)} FASTQ files...")
    
    # Process each FASTQ file
    processed_samples = 0
    for fastq_file in fastq_files:
        try:
            result = processor.process_sample(str(fastq_file))
            if result:
                processed_samples += 1
        except Exception as e:
            print(f"Error processing {fastq_file}: {e}")
    
    if processed_samples > 0:
        # Generate batch report
        print(f"\nGenerating batch report for {processed_samples} samples...")
        processor.generate_batch_report()
        
        print(f"\nQuality Control Analysis Complete!")
        print(f"Processed {processed_samples} samples")
        print(f"Results saved in: {args.output_dir}")
        print(f"  - Individual sample reports: *_qc_report.json")
        print(f"  - Quality plots: *_qc_plots.png")
        print(f"  - Batch summary: batch_qc_report.json")
        print(f"  - HTML report: batch_qc_report.html")
        print(f"  - Batch plots: batch_comparison_plots.png")
    else:
        print("No samples were successfully processed")

if __name__ == "__main__":
    main()

# Example usage:
# Create sample data and run QC:
# python fastqc.py --create_sample

# Process existing FASTQ files:
# python fastqc.py --fastq_files sample1.fastq sample2.fastq

# Process all FASTQ files in a directory:
# python fastqc.py --fastq_dir /path/to/fastq/files

# Create multiple samples and process:
# python fastqc.py --create_sample --num_samples 5

# Install requirements:
# pip install matplotlib seaborn pandas numpy