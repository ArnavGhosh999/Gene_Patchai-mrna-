#!/usr/bin/env python3
"""
Gene Optimizer Pipeline using DNA Chisel
Optimizes codons for host-specific usage, GC content, and motif avoidance
Refines mRNA CDS for optimal expression in human cells
Independent pipeline for therapeutic mRNA design
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from datetime import datetime
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

try:
    from dnachisel import *
    import dnachisel as dc
except ImportError:
    print("DNA Chisel not installed. Install with: pip install dnachisel")
    exit(1)

class GeneOptimizer:
    def __init__(self, reference_fasta, dataset_json, assembly_jsonl, output_dir='gene_optimization_output'):
        """Initialize gene optimization processor"""
        self.reference_fasta = reference_fasta
        self.dataset_json = dataset_json
        self.assembly_jsonl = assembly_jsonl
        self.output_dir = output_dir
        self.reference_info = {}
        self.human_codon_table = {}
        self.optimization_results = {}
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load reference information
        self.load_reference_info()
        self.load_human_codon_usage()
        
        # Create subdirectories
        self.sequences_dir = os.path.join(output_dir, 'optimized_sequences')
        self.analysis_dir = os.path.join(output_dir, 'analysis')
        self.plots_dir = os.path.join(output_dir, 'plots')
        
        for directory in [self.sequences_dir, self.analysis_dir, self.plots_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def load_reference_info(self):
        """Load reference genome information"""
        try:
            # Load dataset catalog
            if os.path.exists(self.dataset_json):
                with open(self.dataset_json, 'r') as f:
                    dataset_data = json.load(f)
                    self.reference_info['dataset'] = dataset_data
                    print(f"Loaded dataset: {dataset_data.get('organism', 'Unknown')}")
            
            # Load assembly report
            if os.path.exists(self.assembly_jsonl):
                with open(self.assembly_jsonl, 'r') as f:
                    assembly_data = json.loads(f.readline())
                    self.reference_info['assembly'] = assembly_data
                    assembly_name = assembly_data.get('assemblyInfo', {}).get('assemblyName', 'Unknown')
                    print(f"Assembly: {assembly_name}")
            
        except Exception as e:
            print(f"Warning: Could not load reference info: {e}")
    
    def load_human_codon_usage(self):
        """Load human codon usage frequencies"""
        # Human codon usage table (frequency per thousand)
        # Data from Kazusa Codon Usage Database for Homo sapiens
        self.human_codon_table = {
            'TTT': 17.6, 'TTC': 20.3, 'TTA': 7.7, 'TTG': 12.9,
            'TCT': 15.2, 'TCC': 17.7, 'TCA': 12.2, 'TCG': 4.4,
            'TAT': 12.2, 'TAC': 15.3, 'TAA': 1.0, 'TAG': 0.8,
            'TGT': 10.6, 'TGC': 12.6, 'TGA': 1.6, 'TGG': 13.2,
            'CTT': 13.2, 'CTC': 19.6, 'CTA': 7.2, 'CTG': 39.6,
            'CCT': 17.5, 'CCC': 19.8, 'CCA': 16.9, 'CCG': 6.9,
            'CAT': 10.9, 'CAC': 15.1, 'CAA': 12.3, 'CAG': 34.2,
            'CGT': 4.5, 'CGC': 10.4, 'CGA': 6.2, 'CGG': 11.4,
            'ATT': 16.0, 'ATC': 20.8, 'ATA': 7.5, 'ATG': 22.0,
            'ACT': 13.1, 'ACC': 18.9, 'ACA': 15.1, 'ACG': 6.1,
            'AAT': 17.0, 'AAC': 19.1, 'AAA': 24.4, 'AAG': 31.9,
            'AGT': 12.1, 'AGC': 19.5, 'AGA': 12.2, 'AGG': 12.0,
            'GTT': 11.0, 'GTC': 14.5, 'GTA': 7.1, 'GTG': 28.1,
            'GCT': 18.4, 'GCC': 27.7, 'GCA': 15.8, 'GCG': 7.4,
            'GAT': 21.8, 'GAC': 25.1, 'GAA': 29.0, 'GAG': 39.6,
            'GGT': 10.8, 'GGC': 22.2, 'GGA': 16.5, 'GGG': 16.5
        }
        
        # Normalize to frequencies (0-1)
        total = sum(self.human_codon_table.values())
        self.human_codon_table = {k: v/total for k, v in self.human_codon_table.items()}
        
        print("Loaded human codon usage table")
    
    def create_sample_sequences(self, num_sequences=5):
        """Create sample mRNA sequences for optimization testing"""
        sample_sequences = []
        
        # Common therapeutic protein sequences (simplified examples)
        sequences = [
            # Example 1: Simplified insulin-like sequence
            "ATGAAATACCTGCTGCCGACCGCTGCTAGCCTGTGTAGCCTGTGCGGGTCGAACCCGCTCCGGGGCCTGCTGGTTGAGATGAAGAACAGGAAGCTTGCCACAAAGGGGAGGCCAGATCCAAACAACGACAAGAGCTTCCACCCACTGGTGGCCGCCCTGATGAGCCAGCTAACTCAGGAACTGCCCCAGGTGCCCGCCACCCGGAGTTCTTATGGCCCGGTGAAGGCGGCCCCGATCGCCCCGATGAAGCCCTGCTACTTGGCCAGGGACGAGCCCTTAGTGCTGGACCGAAGGGGGAGGCCGTAGGGGGCCCCGGCCCTTTTTATAA",
            
            # Example 2: Simplified VEGF-like sequence  
            "ATGAACTTTCTGCTGTCTTGGGTGCATTGGAGCCTTGCCTTGCTGCTCTACCTCCACCATGCCAAGTGGTCCCAGGCTGCACCCATGGCAGAAGGAGGAGGGCAGAATCATCACGAAGTGGTGAAGTTCATGGATGTCTATCAGCGCAGCTACTGCCATCCAATCGAGACCCTGGTGGACATCTTCCAGGAGTACCCTGATGAGATCGAGTACATCTTCAAGCCATCCTGTGTGCCCCTGATGCGATGCGGGGGCTGCTGCAATGACGAGGGCCTGGAGTGTGTGCCCACTGAGGAGTCCAACATCACCATGCAGATTATGCGGATCAAACCTCACCAAGGCCAGCACATAGGAGAGATGAGCTTCCTACAGCACAACAAATGTGAATGCAGACCAAAGAAAGACAAATACAACGTAA",
            
            # Example 3: Simplified antibody light chain-like sequence
            "ATGGATTGGCTGTGGAACCTGGCCCTGTTCCTCCTGTTCCTGGTTGCCACAGGTGCCAGGTCGGAGATCCAGATGACACAGACTACATCCTCCCTGTCTGCCTCTCTGGGAGACAGAGTCACCATCAGTTGCAGGGCAAGTCAGGACATTAGTAAATATTTAAATTGGTATCAGCAGAAACCAGATGGAACTGTTAAACTCCTGATCTACCATACATCAAGATTACACTCAGGAGTCCCATCAAGGTTCAGTGGCAGTGGGTCTGGAACAGATTATTCTCTCACCATTAGCAACCTGGAGCAAGAAGATATTGCCACTTACTTTTGCCAACAGGGTAATACGCTTCCGTACACGTTCGGAGGGGGGACTAAGTTGGAGATCAAACGAACTGTGGCTGCACCATCTGTCTTCATCTTCCCGCCATCTGATGAGCAGTTGAAATCTGGAACTGCCTCTGTTGTGTGCCTGCTGAATAACTTCTATCCCAGAGAGGCCAAAGTACAGTGGAAGGTGGATAACGCCCTCCAATCGGGTAACTCCCAGGAGAGTGTCACAGAGCAGGACAGCAAGGACAGCACCTACAGCCTCAGCAGCACCCTGACGCTGAGCAAAGCAGACTACGAGAAACACAAAGTCTACGCCTGCGAAGTCACCCATCAGGGCCTGAGCTCGCCCGTCACAAAGAGCTTCAACAGGGGAGAGTGTTAG",
            
            # Example 4: Simplified cytokine-like sequence
            "ATGCACAGGCTGCTGGCCCTGCTGCTCCTGGGGGCCCTGCTGCCTGCCGCCGCCACCCCCTCCCTGCCCGAGGCCCGAGACCCACCACCCATGAAGCTCTGCACCTGCACAGCAGATGAGATCTGGAGAATGAAGAACCTGGGGGGCTGCACAGTGGTTGCCAGGGATGTTAACGAGATCTTCCAGAACACCCCAAGGATGCTGGACGGCAAGTTTGACAGCGGCACCACCGAGGTGCTGAGCACATTCACCTCGTCCCTGTTTGACAGGTCGAGCCTGACCACCCCCAGTGTGTGGGACTTCGTGAAGGCCAAGAAGGCACCTTCGGCCGCCGACGTGCTGTGCTGCCTGAGCCTGCGGGAGGCCGCCCACATCCCCGAGGCCCCCACCCCCGAGGCCCCCACCAAGCCCGAGGCCAGGCGGATGA",
            
            # Example 5: Simplified enzyme-like sequence
            "ATGACTAAGCTGCAGGCAATCGCTGCCCTGACCGCTGCCAGCACCTCCCAGGGCACCCCCACCCTGGACACCACCGAGGAGGTGATCAGGAAGATCGCCGAGCGGATCAAGGAGCTGGAGAAGCGGATCAAGGGCCTGACCTTCACCGTGGACAAGGGCGTGGGCGTGGCCCTGCGGGCCACCTGGGGCACCGAGGTGGCCACCGCCTTCGAGGGCGAGATGAAGATCCTGACCGGCGAGCTGGCCGAGGCCCTGGACATCGCCGTGGCCAAGGGCGGCCCCAAGGTGCCCGTGATCGAGGGCCAGTTCCTGGGCACCTTCCCCGCCGTGACCGCCGCCGAGGGCGGCACCGCCACCGTGGCCGGCGGCACCACCGCCACCGTGGCCGGCTGA"
        ]
        
        for i, seq in enumerate(sequences):
            sample_sequences.append({
                'id': f'sample_mrna_{i+1}',
                'name': f'Sample_mRNA_Sequence_{i+1}',
                'sequence': seq,
                'description': f'Sample therapeutic mRNA sequence for optimization testing'
            })
        
        print(f"Created {len(sample_sequences)} sample sequences for optimization")
        return sample_sequences
    
    def extract_cds_from_fasta(self, max_sequences=10):
        """Extract coding sequences from reference FASTA file"""
        try:
            sequences = []
            count = 0
            
            print(f"Extracting sequences from {self.reference_fasta}")
            
            with open(self.reference_fasta, 'r') as fasta_file:
                for record in SeqIO.parse(fasta_file, 'fasta'):
                    if count >= max_sequences:
                        break
                    
                    # Take subsequences from large chromosomes (simulating genes)
                    seq_str = str(record.seq).upper()
                    
                    # Extract multiple CDS-like regions per chromosome
                    for start_pos in range(0, min(len(seq_str), 50000), 10000):
                        if count >= max_sequences:
                            break
                        
                        end_pos = min(start_pos + 1500, len(seq_str))  # ~500 amino acids
                        subseq = seq_str[start_pos:end_pos]
                        
                        # Clean sequence - remove invalid characters
                        valid_bases = set('ATGC')
                        cleaned_seq = ''.join(base if base in valid_bases else 'A' for base in subseq)
                        
                        # Ensure sequence length is divisible by 3
                        cleaned_seq = cleaned_seq[:len(cleaned_seq)//3*3]
                        
                        # Check for minimum length and valid content
                        if len(cleaned_seq) >= 300:  # At least 100 codons
                            # Ensure it doesn't have too many repeats or low complexity
                            gc_content = (cleaned_seq.count('G') + cleaned_seq.count('C')) / len(cleaned_seq)
                            
                            if 0.3 <= gc_content <= 0.7:  # Reasonable GC content
                                sequences.append({
                                    'id': f'CDS_{record.id}_{start_pos}_{end_pos}',
                                    'name': f'CDS from {record.id}',
                                    'sequence': cleaned_seq,
                                    'description': f'Extracted CDS from {record.id} ({start_pos}-{end_pos})'
                                })
                                count += 1
            
            print(f"Extracted {len(sequences)} CDS sequences from reference")
            return sequences
            
        except Exception as e:
            print(f"Error extracting from FASTA: {e}")
            return []
    
    def analyze_sequence_properties(self, sequence):
        """Analyze sequence properties before optimization"""
        seq_str = sequence.upper()
        
        properties = {
            'length': len(seq_str),
            'gc_content': (seq_str.count('G') + seq_str.count('C')) / len(seq_str) * 100,
            'codon_count': len(seq_str) // 3,
        }
        
        # Codon usage analysis
        codons = [seq_str[i:i+3] for i in range(0, len(seq_str), 3)]
        codon_counts = Counter(codons)
        
        # Calculate codon adaptation index (simplified)
        cai_scores = []
        for codon in codons:
            if codon in self.human_codon_table:
                cai_scores.append(self.human_codon_table[codon])
        
        properties['cai_score'] = np.mean(cai_scores) if cai_scores else 0
        properties['rare_codons'] = sum(1 for codon in codons 
                                      if codon in self.human_codon_table and 
                                      self.human_codon_table[codon] < 0.01)
        
        # Check for problematic motifs
        problematic_motifs = ['AATAAA', 'ATTAAA', 'AAGAAG', 'CCGCCC', 'GGGGGG']
        properties['problematic_motifs'] = sum(seq_str.count(motif) for motif in problematic_motifs)
        
        return properties
    
    def design_optimization_constraints(self, sequence, target_gc=50, avoid_motifs=True):
        """Design DNA Chisel constraints for mRNA optimization"""
        constraints = []
        objectives = []
        
        # 1. Preserve amino acid sequence (most important)
        constraints.append(dc.EnforceTranslation())
        
        # 2. GC content optimization (very relaxed range)
        gc_min = max(25, target_gc - 20)  # Very flexible range
        gc_max = min(75, target_gc + 20)
        constraints.append(dc.EnforceGCContent(mini=gc_min/100, maxi=gc_max/100))
        
        # 3. Codon optimization for human expression (primary objective)
        objectives.append(dc.CodonOptimize(species='h_sapiens', method='use_best_codon'))
        
        # 4. Minimal motif avoidance (only essential ones)
        if avoid_motifs:
            # Only avoid the most critical motifs
            constraints.append(dc.AvoidPattern('AATAAA'))  # polyA signal
            constraints.append(dc.AvoidPattern('ATTAAA'))  # polyA signal
        
        # 5. CAI optimization (secondary objective)
        objectives.append(dc.MaximizeCAI(species='h_sapiens'))
        
        return constraints, objectives
    
    def optimize_sequence(self, seq_info, target_gc=50, avoid_motifs=True):
        """Optimize a single sequence using DNA Chisel"""
        try:
            sequence = seq_info['sequence'].upper()
            seq_id = seq_info['id']
            
            print(f"Optimizing {seq_id}...")
            
            # Analyze original sequence
            original_props = self.analyze_sequence_properties(sequence)
            
            # Design constraints and objectives
            constraints, objectives = self.design_optimization_constraints(
                sequence, target_gc, avoid_motifs
            )
            
            # Create DNA Chisel problem
            problem = dc.DnaOptimizationProblem(
                sequence=sequence,
                constraints=constraints,
                objectives=objectives
            )
            
            # Solve the optimization problem
            problem.resolve_constraints()
            problem.optimize()
            
            # Get optimized sequence
            optimized_sequence = problem.sequence
            
            # Analyze optimized sequence
            optimized_props = self.analyze_sequence_properties(optimized_sequence)
            
            # Calculate improvement metrics
            improvements = {
                'gc_content_change': optimized_props['gc_content'] - original_props['gc_content'],
                'cai_improvement': optimized_props['cai_score'] - original_props['cai_score'],
                'rare_codons_reduced': original_props['rare_codons'] - optimized_props['rare_codons'],
                'motifs_removed': original_props['problematic_motifs'] - optimized_props['problematic_motifs']
            }
            
            optimization_result = {
                'sequence_id': seq_id,
                'original_sequence': sequence,
                'optimized_sequence': optimized_sequence,
                'original_properties': original_props,
                'optimized_properties': optimized_props,
                'improvements': improvements,
                'optimization_successful': True,
                'constraints_satisfied': len(problem.constraints_evaluations()) == 0
            }
            
            print(f"  Optimization complete for {seq_id}")
            print(f"  GC content: {original_props['gc_content']:.1f}% → {optimized_props['gc_content']:.1f}%")
            print(f"  CAI score: {original_props['cai_score']:.3f} → {optimized_props['cai_score']:.3f}")
            
            return optimization_result
            
        except Exception as e:
            print(f"Error optimizing {seq_info['id']}: {e}")
            return {
                'sequence_id': seq_info['id'],
                'optimization_successful': False,
                'error': str(e)
            }
    
    def batch_optimize_sequences(self, sequences, target_gc=50, avoid_motifs=True):
        """Optimize multiple sequences in batch"""
        print(f"Starting batch optimization of {len(sequences)} sequences...")
        
        results = []
        successful_optimizations = 0
        
        for i, seq_info in enumerate(sequences, 1):
            print(f"\nProgress: {i}/{len(sequences)}")
            
            result = self.optimize_sequence(seq_info, target_gc, avoid_motifs)
            results.append(result)
            
            if result.get('optimization_successful', False):
                successful_optimizations += 1
        
        print(f"\nBatch optimization complete!")
        print(f"Successful optimizations: {successful_optimizations}/{len(sequences)}")
        
        return results
    
    def generate_optimization_plots(self, results):
        """Generate visualization plots for optimization results"""
        successful_results = [r for r in results if r.get('optimization_successful', False)]
        
        if not successful_results:
            print("No successful optimizations to plot")
            return []
        
        plot_files = []
        
        # 1. GC Content comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Gene Optimization Results', fontsize=16)
        
        # GC content before/after
        original_gc = [r['original_properties']['gc_content'] for r in successful_results]
        optimized_gc = [r['optimized_properties']['gc_content'] for r in successful_results]
        
        axes[0, 0].scatter(original_gc, optimized_gc, alpha=0.6, s=50)
        axes[0, 0].plot([20, 80], [20, 80], 'r--', alpha=0.5)
        axes[0, 0].set_xlabel('Original GC Content (%)')
        axes[0, 0].set_ylabel('Optimized GC Content (%)')
        axes[0, 0].set_title('GC Content Optimization')
        axes[0, 0].grid(True, alpha=0.3)
        
        # CAI score improvement
        original_cai = [r['original_properties']['cai_score'] for r in successful_results]
        optimized_cai = [r['optimized_properties']['cai_score'] for r in successful_results]
        
        axes[0, 1].scatter(original_cai, optimized_cai, alpha=0.6, s=50, color='green')
        axes[0, 1].plot([0, max(max(original_cai), max(optimized_cai))], 
                       [0, max(max(original_cai), max(optimized_cai))], 'r--', alpha=0.5)
        axes[0, 1].set_xlabel('Original CAI Score')
        axes[0, 1].set_ylabel('Optimized CAI Score')
        axes[0, 1].set_title('Codon Adaptation Index Improvement')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Rare codons reduction
        rare_codon_reduction = [r['improvements']['rare_codons_reduced'] for r in successful_results]
        axes[1, 0].hist(rare_codon_reduction, bins=20, alpha=0.7, color='orange')
        axes[1, 0].set_xlabel('Rare Codons Reduced')
        axes[1, 0].set_ylabel('Number of Sequences')
        axes[1, 0].set_title('Rare Codon Reduction Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Overall improvement metrics
        gc_improvements = [abs(r['improvements']['gc_content_change']) for r in successful_results]
        cai_improvements = [r['improvements']['cai_improvement'] for r in successful_results]
        
        x_pos = range(len(successful_results))
        width = 0.35
        
        axes[1, 1].bar([x - width/2 for x in x_pos], gc_improvements, width, 
                      label='GC Content Change', alpha=0.7, color='blue')
        axes[1, 1].bar([x + width/2 for x in x_pos], 
                      [c * 100 for c in cai_improvements], width,
                      label='CAI Improvement (×100)', alpha=0.7, color='red')
        axes[1, 1].set_xlabel('Sequence Index')
        axes[1, 1].set_ylabel('Improvement Magnitude')
        axes[1, 1].set_title('Optimization Improvements by Sequence')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        optimization_plot = os.path.join(self.plots_dir, 'optimization_results.png')
        plt.savefig(optimization_plot, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(optimization_plot)
        
        # 2. Detailed sequence analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Detailed Sequence Analysis', fontsize=16)
        
        # Sequence length distribution
        lengths = [r['original_properties']['length'] for r in successful_results]
        axes[0, 0].hist(lengths, bins=15, alpha=0.7, color='purple')
        axes[0, 0].set_xlabel('Sequence Length (bp)')
        axes[0, 0].set_ylabel('Number of Sequences')
        axes[0, 0].set_title('Sequence Length Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # GC content distribution comparison
        axes[0, 1].hist(original_gc, bins=15, alpha=0.5, label='Original', color='red')
        axes[0, 1].hist(optimized_gc, bins=15, alpha=0.5, label='Optimized', color='green')
        axes[0, 1].set_xlabel('GC Content (%)')
        axes[0, 1].set_ylabel('Number of Sequences')
        axes[0, 1].set_title('GC Content Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Motif removal effectiveness
        motifs_removed = [r['improvements']['motifs_removed'] for r in successful_results]
        axes[1, 0].bar(range(len(motifs_removed)), motifs_removed, alpha=0.7, color='orange')
        axes[1, 0].set_xlabel('Sequence Index')
        axes[1, 0].set_ylabel('Problematic Motifs Removed')
        axes[1, 0].set_title('Motif Removal Effectiveness')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Optimization success rate
        total_sequences = len(results)
        successful_sequences = len(successful_results)
        
        labels = ['Successful', 'Failed']
        sizes = [successful_sequences, total_sequences - successful_sequences]
        colors = ['lightgreen', 'lightcoral']
        
        axes[1, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('Optimization Success Rate')
        
        plt.tight_layout()
        detailed_plot = os.path.join(self.plots_dir, 'detailed_analysis.png')
        plt.savefig(detailed_plot, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(detailed_plot)
        
        print(f"Generated {len(plot_files)} optimization plots")
        return plot_files
    
    def save_optimized_sequences(self, results):
        """Save optimized sequences in FASTA format"""
        successful_results = [r for r in results if r.get('optimization_successful', False)]
        
        if not successful_results:
            print("No successful optimizations to save")
            return None
        
        # Save original sequences
        original_fasta = os.path.join(self.sequences_dir, 'original_sequences.fasta')
        with open(original_fasta, 'w') as f:
            for result in successful_results:
                f.write(f">{result['sequence_id']}_original\n")
                f.write(f"{result['original_sequence']}\n")
        
        # Save optimized sequences
        optimized_fasta = os.path.join(self.sequences_dir, 'optimized_sequences.fasta')
        with open(optimized_fasta, 'w') as f:
            for result in successful_results:
                f.write(f">{result['sequence_id']}_optimized\n")
                f.write(f"{result['optimized_sequence']}\n")
        
        print(f"Saved sequences:")
        print(f"  Original: {original_fasta}")
        print(f"  Optimized: {optimized_fasta}")
        
        return original_fasta, optimized_fasta
    
    def calculate_summary_statistics(self, results):
        """Calculate summary statistics for optimization results"""
        successful_results = [r for r in results if r.get('optimization_successful', False)]
        
        summary = {
            'successful_optimizations': len(successful_results),
            'total_sequences': len(results),
            'success_rate_percent': (len(successful_results) / len(results)) * 100 if results else 0
        }
        
        if not successful_results:
            summary.update({
                'average_improvements': {
                    'gc_content_change': 0.0,
                    'cai_improvement': 0.0,
                    'rare_codons_reduced': 0.0,
                    'motifs_removed': 0.0
                },
                'gc_content_ranges': {
                    'original': [0, 0],
                    'optimized': [0, 0]
                },
                'sequences_with_improvements': {
                    'gc_content': 0,
                    'cai_score': 0,
                    'rare_codons': 0,
                    'motif_removal': 0
                }
            })
            return summary
        
        # Calculate averages
        avg_gc_change = np.mean([r['improvements']['gc_content_change'] for r in successful_results])
        avg_cai_improvement = np.mean([r['improvements']['cai_improvement'] for r in successful_results])
        avg_rare_codons_reduced = np.mean([r['improvements']['rare_codons_reduced'] for r in successful_results])
        avg_motifs_removed = np.mean([r['improvements']['motifs_removed'] for r in successful_results])
        
        # Calculate ranges
        original_gc_range = [
            min(r['original_properties']['gc_content'] for r in successful_results),
            max(r['original_properties']['gc_content'] for r in successful_results)
        ]
        optimized_gc_range = [
            min(r['optimized_properties']['gc_content'] for r in successful_results),
            max(r['optimized_properties']['gc_content'] for r in successful_results)
        ]
        
        summary.update({
            'average_improvements': {
                'gc_content_change': float(avg_gc_change),
                'cai_improvement': float(avg_cai_improvement),
                'rare_codons_reduced': float(avg_rare_codons_reduced),
                'motifs_removed': float(avg_motifs_removed)
            },
            'gc_content_ranges': {
                'original': original_gc_range,
                'optimized': optimized_gc_range
            },
            'sequences_with_improvements': {
                'gc_content': sum(1 for r in successful_results if abs(r['improvements']['gc_content_change']) > 1),
                'cai_score': sum(1 for r in successful_results if r['improvements']['cai_improvement'] > 0.01),
                'rare_codons': sum(1 for r in successful_results if r['improvements']['rare_codons_reduced'] > 0),
                'motif_removal': sum(1 for r in successful_results if r['improvements']['motifs_removed'] > 0)
            }
        })
        
        return summary
    
    def create_optimization_report(self, results, report_path):
        """Create detailed optimization report"""
        successful_results = [r for r in results if r.get('optimization_successful', False)]
        summary_stats = self.calculate_summary_statistics(results)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("mRNA Gene Optimization Report\n")
            f.write("=" * 35 + "\n\n")
            
            f.write("OVERVIEW\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total sequences processed: {len(results)}\n")
            f.write(f"Successful optimizations: {len(successful_results)}\n")
            f.write(f"Success rate: {summary_stats['success_rate_percent']:.1f}%\n")
            f.write(f"Reference genome: {os.path.basename(self.reference_fasta)}\n\n")
            
            if self.reference_info.get('dataset'):
                organism = self.reference_info['dataset'].get('organism', 'Unknown')
                f.write(f"Source organism: {organism}\n")
            if self.reference_info.get('assembly'):
                assembly = self.reference_info['assembly'].get('assemblyInfo', {}).get('assemblyName', 'Unknown')
                f.write(f"Assembly: {assembly}\n\n")
            
            f.write("OPTIMIZATION TARGETS\n")
            f.write("-" * 20 + "\n")
            f.write("• Human codon usage optimization\n")
            f.write("• GC content optimization (40-60%)\n")
            f.write("• Rare codon minimization\n")
            f.write("• Problematic motif removal\n")
            f.write("• Restriction site avoidance\n\n")
            
            if successful_results:
                f.write("OPTIMIZATION RESULTS\n")
                f.write("-" * 20 + "\n")
                avg_improvements = summary_stats['average_improvements']
                
                f.write(f"Average GC content change: {avg_improvements['gc_content_change']:+.1f}%\n")
                f.write(f"Average CAI improvement: {avg_improvements['cai_improvement']:+.3f}\n")
                f.write(f"Average rare codons reduced: {avg_improvements['rare_codons_reduced']:.1f}\n")
                f.write(f"Average motifs removed: {avg_improvements['motifs_removed']:.1f}\n\n")
                
                gc_ranges = summary_stats['gc_content_ranges']
                f.write(f"GC content range (original): {gc_ranges['original'][0]:.1f}% - {gc_ranges['original'][1]:.1f}%\n")
                f.write(f"GC content range (optimized): {gc_ranges['optimized'][0]:.1f}% - {gc_ranges['optimized'][1]:.1f}%\n\n")
                
                improvements = summary_stats['sequences_with_improvements']
                f.write("SEQUENCES WITH SIGNIFICANT IMPROVEMENTS\n")
                f.write("-" * 40 + "\n")
                f.write(f"GC content improved: {improvements['gc_content']}/{len(successful_results)}\n")
                f.write(f"CAI score improved: {improvements['cai_score']}/{len(successful_results)}\n")
                f.write(f"Rare codons reduced: {improvements['rare_codons']}/{len(successful_results)}\n")
                f.write(f"Motifs removed: {improvements['motif_removal']}/{len(successful_results)}\n\n")
                
                f.write("TOP OPTIMIZATION RESULTS\n")
                f.write("-" * 30 + "\n")
                
                # Sort by overall improvement score
                scored_results = []
                for result in successful_results:
                    score = (
                        abs(result['improvements']['gc_content_change']) * 0.3 +
                        result['improvements']['cai_improvement'] * 100 * 0.4 +
                        result['improvements']['rare_codons_reduced'] * 0.2 +
                        result['improvements']['motifs_removed'] * 0.1
                    )
                    scored_results.append((score, result))
                
                # Sort by score (first element of tuple)
                scored_results.sort(key=lambda x: x[0], reverse=True)
                
                for i, (score, result) in enumerate(scored_results[:5], 1):
                    f.write(f"\n{i}. {result['sequence_id']}\n")
                    f.write(f"   Overall improvement score: {score:.2f}\n")
                    f.write(f"   GC content: {result['original_properties']['gc_content']:.1f}% to ")
                    f.write(f"{result['optimized_properties']['gc_content']:.1f}% ")
                    f.write(f"({result['improvements']['gc_content_change']:+.1f}%)\n")
                    f.write(f"   CAI score: {result['original_properties']['cai_score']:.3f} to ")
                    f.write(f"{result['optimized_properties']['cai_score']:.3f} ")
                    f.write(f"({result['improvements']['cai_improvement']:+.3f})\n")
                    f.write(f"   Rare codons reduced: {result['improvements']['rare_codons_reduced']}\n")
                    f.write(f"   Motifs removed: {result['improvements']['motifs_removed']}\n")
            
            f.write("\nRECOMMENDATIONS FOR mRNA THERAPY\n")
            f.write("-" * 35 + "\n")
            f.write("• Use optimized sequences for improved human cell expression\n")
            f.write("• Consider further 5' and 3' UTR optimization\n")
            f.write("• Validate expression levels in relevant cell lines\n")
            f.write("• Monitor for immunogenicity with optimized sequences\n")
            f.write("• Consider tissue-specific codon preferences if applicable\n")
    
    def save_results(self, results):
        """Save all optimization results and analysis"""
        # Save detailed results
        results_path = os.path.join(self.analysis_dir, 'gene_optimization_results.json')
        
        # Prepare results for JSON serialization
        json_results = []
        for result in results:
            json_result = result.copy()
            # Convert numpy types to Python types
            for key in ['original_properties', 'optimized_properties', 'improvements']:
                if key in json_result:
                    for prop_key, prop_val in json_result[key].items():
                        if isinstance(prop_val, (np.integer, np.floating)):
                            json_result[key][prop_key] = float(prop_val)
            json_results.append(json_result)
        
        with open(results_path, 'w') as f:
            json.dump({
                'optimization_results': json_results,
                'summary_statistics': self.calculate_summary_statistics(results),
                'reference_info': self.reference_info,
                'processing_timestamp': datetime.now().isoformat()
            }, f, indent=2, default=str)
        
        # Create optimization report
        report_path = os.path.join(self.analysis_dir, 'gene_optimization_report.txt')
        self.create_optimization_report(results, report_path)
        
        print(f"Results saved:")
        print(f"  Detailed results: {results_path}")
        print(f"  Optimization report: {report_path}")
        
        return results_path, report_path

def main():
    parser = argparse.ArgumentParser(description='Gene Optimization Pipeline using DNA Chisel')
    parser.add_argument('--reference', default=r'ncbi_dataset\ncbi_dataset\data\GCA_000001405.29_GRCh38.p14_genomic.fna',
                       help='Reference FASTA file')
    parser.add_argument('--dataset_json', default=r'ncbi_dataset\ncbi_dataset\data\dataset_catalog.json',
                       help='Dataset catalog JSON file')
    parser.add_argument('--assembly_jsonl', default=r'ncbi_dataset\ncbi_dataset\data\assembly_data_report.jsonl',
                       help='Assembly report JSONL file')
    parser.add_argument('--create_sample', action='store_true', help='Create sample sequences for testing')
    parser.add_argument('--use_reference', action='store_true', help='Extract sequences from reference FASTA')
    parser.add_argument('--input_fasta', help='Input FASTA file with sequences to optimize')
    parser.add_argument('--output_dir', default='gene_optimization_output', help='Output directory')
    parser.add_argument('--target_gc', type=float, default=50, help='Target GC content percentage')
    parser.add_argument('--max_sequences', type=int, default=10, help='Maximum sequences to process')
    parser.add_argument('--avoid_motifs', action='store_true', default=True, help='Avoid problematic motifs')
    
    args = parser.parse_args()
    
    # Initialize processor
    optimizer = GeneOptimizer(
        reference_fasta=args.reference,
        dataset_json=args.dataset_json,
        assembly_jsonl=args.assembly_jsonl,
        output_dir=args.output_dir
    )
    
    # Get sequences to optimize
    sequences = []
    
    if args.create_sample:
        print("Creating sample mRNA sequences for optimization...")
        sequences = optimizer.create_sample_sequences()
    elif args.use_reference:
        print("Extracting sequences from reference genome...")
        sequences = optimizer.extract_cds_from_fasta(max_sequences=args.max_sequences)
    elif args.input_fasta:
        print(f"Loading sequences from {args.input_fasta}")
        sequences = []
        with open(args.input_fasta, 'r') as f:
            for record in SeqIO.parse(f, 'fasta'):
                sequences.append({
                    'id': record.id,
                    'name': record.description,
                    'sequence': str(record.seq),
                    'description': record.description
                })
    else:
        print("Error: Specify input sequences using one of:")
        print("  --create_sample (generate test sequences)")
        print("  --use_reference (extract from reference genome)")
        print("  --input_fasta file.fasta (use custom FASTA file)")
        return
    
    if not sequences:
        print("No sequences found for optimization")
        return
    
    print(f"\nStarting gene optimization pipeline...")
    print(f"Sequences to optimize: {len(sequences)}")
    print(f"Target GC content: {args.target_gc}%")
    print(f"Avoid problematic motifs: {args.avoid_motifs}")
    
    # Perform batch optimization
    optimization_results = optimizer.batch_optimize_sequences(
        sequences, 
        target_gc=args.target_gc, 
        avoid_motifs=args.avoid_motifs
    )
    
    # Generate visualizations
    plot_files = optimizer.generate_optimization_plots(optimization_results)
    
    # Save optimized sequences
    fasta_files = optimizer.save_optimized_sequences(optimization_results)
    
    # Save results and generate reports
    results_path, report_path = optimizer.save_results(optimization_results)
    
    # Print summary
    successful_optimizations = sum(1 for r in optimization_results if r.get('optimization_successful', False))
    
    print(f"\nGene Optimization Complete!")
    print(f"Results saved in: {args.output_dir}")
    print(f"Successful optimizations: {successful_optimizations}/{len(sequences)}")
    
    if successful_optimizations > 0:
        print(f"Output files:")
        print(f"  - Optimization plots: {len(plot_files)} files")
        print(f"  - Optimized sequences: {fasta_files}")
        print(f"  - Detailed results: {results_path}")
        print(f"  - Optimization report: {report_path}")
        
        # Calculate and display average improvements
        successful_results = [r for r in optimization_results if r.get('optimization_successful', False)]
        if successful_results:
            avg_gc_change = np.mean([r['improvements']['gc_content_change'] for r in successful_results])
            avg_cai_improvement = np.mean([r['improvements']['cai_improvement'] for r in successful_results])
            avg_rare_reduced = np.mean([r['improvements']['rare_codons_reduced'] for r in successful_results])
            
            print(f"\nAverage Improvements:")
            print(f"  GC content change: {avg_gc_change:+.1f}%")
            print(f"  CAI improvement: {avg_cai_improvement:+.3f}")
            print(f"  Rare codons reduced: {avg_rare_reduced:.1f}")

if __name__ == "__main__":
    main()

# Example usage:
# Create sample sequences and optimize:
# python gene_optimizer.py --create_sample

# Extract from reference genome and optimize:
# python gene_optimizer.py --use_reference --max_sequences 5

# Optimize custom sequences:
# python gene_optimizer.py --input_fasta input_sequences.fasta

# Optimize with specific GC target:
# python gene_optimizer.py --create_sample --target_gc 45

# Install requirements:
# pip install dnachisel biopython matplotlib seaborn pandas numpy