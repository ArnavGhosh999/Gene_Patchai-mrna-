#!/usr/bin/env python3
"""
PacBio SMRT Analysis Pipeline using RDKit
Simulates Single Molecule Real-Time sequencing analysis for HiFi reads
Performs variant detection and transcript discovery on long reads
Independent pipeline for high-accuracy assemblies and rare variant detection
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from datetime import datetime
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
    RDKIT_AVAILABLE = True
    print("RDKit loaded successfully")
except ImportError:
    print("RDKit not installed. Install with: pip install rdkit")
    RDKIT_AVAILABLE = False

import matplotlib.pyplot as plt
import seaborn as sns

class PacBioSMRTProcessor:
    def __init__(self, output_dir='pacbio_smrt_output'):
        """Initialize PacBio SMRT processor"""
        self.output_dir = output_dir
        self.hifi_reads = []
        self.variants = []
        self.transcripts = []
        self.molecular_analysis = {}
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        self.reads_dir = os.path.join(output_dir, 'hifi_reads')
        self.variants_dir = os.path.join(output_dir, 'variants')
        self.transcripts_dir = os.path.join(output_dir, 'transcripts')
        self.analysis_dir = os.path.join(output_dir, 'analysis')
        self.plots_dir = os.path.join(output_dir, 'plots')
        
        for directory in [self.reads_dir, self.variants_dir, self.transcripts_dir, 
                         self.analysis_dir, self.plots_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # SMRT sequencing parameters
        self.smrt_params = {
            'read_length_range': (5000, 25000),  # HiFi read lengths
            'accuracy_threshold': 0.99,  # HiFi accuracy
            'pass_threshold': 3,  # CCS passes
            'polymerase_speed': 2.5,  # bp/second
            'movie_time': 1800  # 30 minutes
        }
        
        # Molecular patterns for variant analysis
        self.nucleotide_patterns = {
            'A': 'C5H5N5',  # Adenine
            'T': 'C5H6N2O2',  # Thymine  
            'G': 'C5H5N5O',  # Guanine
            'C': 'C4H5N3O'   # Cytosine
        }
    
    def create_sample_hifi_reads(self, num_reads=1000, target_genes=None):
        """Create sample HiFi reads for testing"""
        print(f"Generating {num_reads} sample HiFi reads...")
        
        if target_genes is None:
            target_genes = [
                'BRCA1', 'BRCA2', 'TP53', 'EGFR', 'KRAS', 
                'PIK3CA', 'APC', 'PTEN', 'RB1', 'VHL'
            ]
        
        hifi_reads = []
        
        for i in range(num_reads):
            # Generate realistic HiFi read
            read_length = np.random.randint(
                self.smrt_params['read_length_range'][0],
                self.smrt_params['read_length_range'][1]
            )
            
            # Select target gene
            target_gene = np.random.choice(target_genes)
            
            # Generate sequence with potential variants
            sequence, variants = self.generate_sequence_with_variants(read_length, target_gene)
            
            # Calculate quality metrics
            accuracy = np.random.normal(0.995, 0.003)  # HiFi accuracy
            accuracy = max(0.99, min(0.999, accuracy))
            
            ccs_passes = np.random.poisson(5) + 3  # CCS passes
            
            read_info = {
                'read_id': f'HiFi_read_{i:06d}',
                'sequence': sequence,
                'length': read_length,
                'accuracy': accuracy,
                'ccs_passes': ccs_passes,
                'target_gene': target_gene,
                'variants': variants,
                'gc_content': self.calculate_gc_content(sequence),
                'molecular_weight': self.calculate_molecular_weight(sequence),
                'timestamp': datetime.now().isoformat()
            }
            
            hifi_reads.append(read_info)
        
        self.hifi_reads = hifi_reads
        print(f"Generated {len(hifi_reads)} HiFi reads")
        return hifi_reads
    
    def generate_sequence_with_variants(self, length, gene_name):
        """Generate DNA sequence with embedded variants"""
        # Generate base sequence
        bases = ['A', 'T', 'G', 'C']
        sequence = ''.join(np.random.choice(bases, length))
        
        # Introduce variants based on gene type
        variants = []
        variant_positions = []
        
        # SNP rate (1 per 300-1000 bp for rare variants)
        snp_rate = 1 / np.random.randint(300, 1000)
        num_snps = int(length * snp_rate)
        
        for _ in range(num_snps):
            pos = np.random.randint(0, length)
            if pos not in variant_positions:
                original_base = sequence[pos]
                new_base = np.random.choice([b for b in bases if b != original_base])
                
                # Replace base in sequence
                sequence = sequence[:pos] + new_base + sequence[pos+1:]
                
                variant_info = {
                    'type': 'SNP',
                    'position': pos,
                    'ref': original_base,
                    'alt': new_base,
                    'gene': gene_name,
                    'frequency': np.random.uniform(0.01, 0.5)  # Allele frequency
                }
                
                variants.append(variant_info)
                variant_positions.append(pos)
        
        # Add some indels (1 per 5-10 kb)
        indel_rate = 1 / np.random.randint(5000, 10000)
        num_indels = max(1, int(length * indel_rate))
        
        for _ in range(num_indels):
            pos = np.random.randint(0, length - 10)
            if pos not in variant_positions:
                indel_type = np.random.choice(['insertion', 'deletion'])
                indel_size = np.random.randint(1, 5)
                
                if indel_type == 'insertion':
                    insert_seq = ''.join(np.random.choice(bases, indel_size))
                    sequence = sequence[:pos] + insert_seq + sequence[pos:]
                    
                    variant_info = {
                        'type': 'INS',
                        'position': pos,
                        'ref': '',
                        'alt': insert_seq,
                        'gene': gene_name,
                        'size': indel_size
                    }
                else:  # deletion
                    if pos + indel_size < len(sequence):
                        deleted_seq = sequence[pos:pos + indel_size]
                        sequence = sequence[:pos] + sequence[pos + indel_size:]
                        
                        variant_info = {
                            'type': 'DEL',
                            'position': pos,
                            'ref': deleted_seq,
                            'alt': '',
                            'gene': gene_name,
                            'size': indel_size
                        }
                
                variants.append(variant_info)
                variant_positions.append(pos)
        
        return sequence, variants
    
    def calculate_gc_content(self, sequence):
        """Calculate GC content of sequence"""
        gc_count = sequence.count('G') + sequence.count('C')
        return (gc_count / len(sequence)) * 100 if sequence else 0
    
    def calculate_molecular_weight(self, sequence):
        """Calculate molecular weight using RDKit-style analysis"""
        if not RDKIT_AVAILABLE:
            # Simplified calculation without RDKit
            weights = {'A': 331.2, 'T': 322.2, 'G': 347.2, 'C': 307.2}
            return sum(weights.get(base, 325) for base in sequence)
        
        # Use RDKit for more sophisticated molecular analysis
        try:
            # Convert DNA sequence to molecular representation
            total_weight = 0
            for base in sequence:
                if base in self.nucleotide_patterns:
                    mol_formula = self.nucleotide_patterns[base]
                    mol = Chem.MolFromSmiles(self.formula_to_smiles(mol_formula))
                    if mol:
                        total_weight += Descriptors.MolWt(mol)
                    else:
                        # Fallback weights
                        weights = {'A': 331.2, 'T': 322.2, 'G': 347.2, 'C': 307.2}
                        total_weight += weights.get(base, 325)
            
            return total_weight
            
        except Exception as e:
            # Fallback calculation
            weights = {'A': 331.2, 'T': 322.2, 'G': 347.2, 'C': 307.2}
            return sum(weights.get(base, 325) for base in sequence)
    
    def formula_to_smiles(self, formula):
        """Convert molecular formula to SMILES (simplified)"""
        # Simplified conversion for nucleotides
        smiles_map = {
            'C5H5N5': 'C1=NC(=C2C(=N1)N=CN2)N',  # Adenine
            'C5H6N2O2': 'CC1=CN(C(=O)NC1=O)',     # Thymine
            'C5H5N5O': 'C1=NC2=C(N1)C(=O)N=CN2',  # Guanine
            'C4H5N3O': 'C1=CN(C(=O)N=C1)N'        # Cytosine
        }
        return smiles_map.get(formula, 'C')
    
    def analyze_molecular_properties(self, sequence):
        """Analyze molecular properties using RDKit"""
        if not RDKIT_AVAILABLE:
            return {'error': 'RDKit not available'}
        
        try:
            # Convert sequence to molecular representation for analysis
            properties = {
                'length': len(sequence),
                'gc_content': self.calculate_gc_content(sequence),
                'molecular_weight': self.calculate_molecular_weight(sequence),
                'purine_content': (sequence.count('A') + sequence.count('G')) / len(sequence) * 100,
                'pyrimidine_content': (sequence.count('T') + sequence.count('C')) / len(sequence) * 100
            }
            
            # Calculate dinucleotide frequencies
            dinucleotides = {}
            for i in range(len(sequence) - 1):
                dinuc = sequence[i:i+2]
                dinucleotides[dinuc] = dinucleotides.get(dinuc, 0) + 1
            
            properties['dinucleotide_frequencies'] = dinucleotides
            
            # Complexity analysis
            properties['complexity_score'] = self.calculate_complexity(sequence)
            
            return properties
            
        except Exception as e:
            return {'error': str(e)}
    
    def calculate_complexity(self, sequence):
        """Calculate sequence complexity"""
        # Measure sequence complexity using k-mer diversity
        k = 3  # tri-nucleotides
        kmers = set()
        
        for i in range(len(sequence) - k + 1):
            kmers.add(sequence[i:i+k])
        
        # Complexity score: unique k-mers / total possible
        total_kmers = len(sequence) - k + 1
        complexity = len(kmers) / total_kmers if total_kmers > 0 else 0
        
        return complexity
    
    def detect_variants(self, reference_sequence=None):
        """Detect variants in HiFi reads"""
        print("Detecting variants in HiFi reads...")
        
        all_variants = []
        variant_stats = defaultdict(int)
        
        for read in self.hifi_reads:
            for variant in read['variants']:
                # Add read context to variant
                variant_with_context = variant.copy()
                variant_with_context['read_id'] = read['read_id']
                variant_with_context['read_accuracy'] = read['accuracy']
                variant_with_context['read_length'] = read['length']
                
                # Quality scoring
                quality_score = self.calculate_variant_quality(variant, read)
                variant_with_context['quality_score'] = quality_score
                
                # Pathogenicity prediction (simplified)
                pathogenicity = self.predict_pathogenicity(variant)
                variant_with_context['pathogenicity'] = pathogenicity
                
                all_variants.append(variant_with_context)
                variant_stats[variant['type']] += 1
        
        # Filter high-quality variants
        high_quality_variants = [
            v for v in all_variants 
            if v['quality_score'] > 30 and v['read_accuracy'] > 0.99
        ]
        
        self.variants = high_quality_variants
        
        print(f"Detected {len(all_variants)} total variants")
        print(f"High-quality variants: {len(high_quality_variants)}")
        print(f"Variant types: {dict(variant_stats)}")
        
        return high_quality_variants
    
    def calculate_variant_quality(self, variant, read):
        """Calculate variant quality score"""
        # Quality based on read accuracy, CCS passes, and variant characteristics
        base_quality = read['accuracy'] * 100
        ccs_bonus = min(read['ccs_passes'] * 2, 20)
        
        # Position penalty (variants at read ends are less reliable)
        position_penalty = 0
        read_length = read['length']
        position = variant['position']
        
        if position < read_length * 0.1 or position > read_length * 0.9:
            position_penalty = 5
        
        quality_score = base_quality + ccs_bonus - position_penalty
        return max(0, min(60, quality_score))  # Cap at 60
    
    def predict_pathogenicity(self, variant):
        """Predict variant pathogenicity (simplified)"""
        # Simplified pathogenicity prediction
        if variant['type'] == 'SNP':
            # Check if in known pathogenic positions (simplified)
            if variant['gene'] in ['BRCA1', 'BRCA2', 'TP53']:
                return 'Likely Pathogenic' if np.random.random() > 0.7 else 'Uncertain'
            else:
                return 'Likely Benign' if np.random.random() > 0.3 else 'Uncertain'
        
        elif variant['type'] in ['INS', 'DEL']:
            # Indels are more likely to be pathogenic
            size = variant.get('size', 1)
            if size >= 3:  # Affects codon
                return 'Likely Pathogenic' if np.random.random() > 0.5 else 'Uncertain'
            else:
                return 'Uncertain'
        
        return 'Uncertain'
    
    def discover_transcripts(self):
        """Discover transcript isoforms from HiFi reads"""
        print("Discovering transcript isoforms...")
        
        # Group reads by target gene
        gene_reads = defaultdict(list)
        for read in self.hifi_reads:
            gene_reads[read['target_gene']].append(read)
        
        transcripts = []
        
        for gene, reads in gene_reads.items():
            # Analyze splice patterns and isoforms
            isoforms = self.identify_isoforms(reads, gene)
            
            for isoform in isoforms:
                transcript_info = {
                    'transcript_id': f"{gene}_isoform_{len(transcripts) + 1}",
                    'gene_name': gene,
                    'isoform_type': isoform['type'],
                    'supporting_reads': len(isoform['reads']),
                    'average_length': np.mean([r['length'] for r in isoform['reads']]),
                    'expression_level': self.estimate_expression_level(isoform['reads']),
                    'splice_junctions': isoform.get('splice_junctions', []),
                    'molecular_properties': self.analyze_isoform_properties(isoform['reads'])
                }
                
                transcripts.append(transcript_info)
        
        self.transcripts = transcripts
        
        print(f"Discovered {len(transcripts)} transcript isoforms")
        for gene in gene_reads.keys():
            gene_transcripts = [t for t in transcripts if t['gene_name'] == gene]
            print(f"  {gene}: {len(gene_transcripts)} isoforms")
        
        return transcripts
    
    def identify_isoforms(self, reads, gene):
        """Identify transcript isoforms from reads"""
        # Simplified isoform identification
        # Group reads by length ranges to identify different isoforms
        
        length_groups = defaultdict(list)
        for read in reads:
            # Group by length bins
            length_bin = (read['length'] // 1000) * 1000
            length_groups[length_bin].append(read)
        
        isoforms = []
        
        for length_bin, grouped_reads in length_groups.items():
            if len(grouped_reads) >= 3:  # Minimum support
                # Determine isoform type based on length
                if length_bin < 10000:
                    isoform_type = 'short'
                elif length_bin > 20000:
                    isoform_type = 'long'
                else:
                    isoform_type = 'canonical'
                
                # Simulate splice junctions
                splice_junctions = []
                if np.random.random() > 0.5:  # 50% chance of alternative splicing
                    num_junctions = np.random.randint(1, 4)
                    for _ in range(num_junctions):
                        junction = {
                            'donor': np.random.randint(1000, length_bin - 1000),
                            'acceptor': np.random.randint(1000, length_bin - 1000),
                            'type': np.random.choice(['exon_skipping', 'intron_retention', 'alt_splice_site'])
                        }
                        splice_junctions.append(junction)
                
                isoform = {
                    'type': isoform_type,
                    'reads': grouped_reads,
                    'splice_junctions': splice_junctions
                }
                
                isoforms.append(isoform)
        
        return isoforms
    
    def estimate_expression_level(self, reads):
        """Estimate expression level from read support"""
        # Simple expression level based on read count and quality
        read_count = len(reads)
        avg_accuracy = np.mean([r['accuracy'] for r in reads])
        
        # Normalize expression level
        expression_level = read_count * avg_accuracy * 100
        
        return expression_level
    
    def analyze_isoform_properties(self, reads):
        """Analyze molecular properties of transcript isoforms"""
        # Combine sequences for analysis
        combined_sequence = ''.join([r['sequence'] for r in reads[:3]])  # Sample reads
        
        return self.analyze_molecular_properties(combined_sequence)
    
    def generate_analysis_plots(self):
        """Generate comprehensive analysis plots"""
        if not self.hifi_reads:
            print("No HiFi reads available for plotting")
            return []
        
        plot_files = []
        
        # 1. Read length and accuracy distribution
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('PacBio SMRT HiFi Analysis', fontsize=16)
        
        # Read length distribution
        lengths = [read['length'] for read in self.hifi_reads]
        axes[0, 0].hist(lengths, bins=30, alpha=0.7, color='blue')
        axes[0, 0].set_xlabel('Read Length (bp)')
        axes[0, 0].set_ylabel('Number of Reads')
        axes[0, 0].set_title('HiFi Read Length Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy distribution
        accuracies = [read['accuracy'] * 100 for read in self.hifi_reads]
        axes[0, 1].hist(accuracies, bins=20, alpha=0.7, color='green')
        axes[0, 1].set_xlabel('Accuracy (%)')
        axes[0, 1].set_ylabel('Number of Reads')
        axes[0, 1].set_title('HiFi Read Accuracy Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # GC content distribution
        gc_contents = [read['gc_content'] for read in self.hifi_reads]
        axes[1, 0].hist(gc_contents, bins=25, alpha=0.7, color='orange')
        axes[1, 0].set_xlabel('GC Content (%)')
        axes[1, 0].set_ylabel('Number of Reads')
        axes[1, 0].set_title('GC Content Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Gene distribution
        genes = [read['target_gene'] for read in self.hifi_reads]
        gene_counts = Counter(genes)
        axes[1, 1].bar(gene_counts.keys(), gene_counts.values(), alpha=0.7, color='purple')
        axes[1, 1].set_xlabel('Target Genes')
        axes[1, 1].set_ylabel('Number of Reads')
        axes[1, 1].set_title('Reads per Target Gene')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        reads_plot = os.path.join(self.plots_dir, 'hifi_reads_analysis.png')
        plt.savefig(reads_plot, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(reads_plot)
        
        # 2. Variant analysis plots
        if self.variants:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Variant Detection Analysis', fontsize=16)
            
            # Variant types
            variant_types = [v['type'] for v in self.variants]
            type_counts = Counter(variant_types)
            axes[0, 0].pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%')
            axes[0, 0].set_title('Variant Types Distribution')
            
            # Quality scores
            quality_scores = [v['quality_score'] for v in self.variants]
            axes[0, 1].hist(quality_scores, bins=20, alpha=0.7, color='red')
            axes[0, 1].set_xlabel('Quality Score')
            axes[0, 1].set_ylabel('Number of Variants')
            axes[0, 1].set_title('Variant Quality Distribution')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Pathogenicity prediction
            pathogenicity = [v['pathogenicity'] for v in self.variants]
            path_counts = Counter(pathogenicity)
            axes[1, 0].bar(path_counts.keys(), path_counts.values(), alpha=0.7, color='coral')
            axes[1, 0].set_xlabel('Pathogenicity')
            axes[1, 0].set_ylabel('Number of Variants')
            axes[1, 0].set_title('Pathogenicity Predictions')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
            
            # Variants per gene
            variant_genes = [v['gene'] for v in self.variants]
            gene_var_counts = Counter(variant_genes)
            axes[1, 1].bar(gene_var_counts.keys(), gene_var_counts.values(), alpha=0.7, color='teal')
            axes[1, 1].set_xlabel('Genes')
            axes[1, 1].set_ylabel('Number of Variants')
            axes[1, 1].set_title('Variants per Gene')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            variants_plot = os.path.join(self.plots_dir, 'variant_analysis.png')
            plt.savefig(variants_plot, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files.append(variants_plot)
        
        # 3. Transcript discovery plots
        if self.transcripts:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Transcript Discovery Analysis', fontsize=16)
            
            # Expression levels
            expression_levels = [t['expression_level'] for t in self.transcripts]
            axes[0, 0].hist(expression_levels, bins=20, alpha=0.7, color='green')
            axes[0, 0].set_xlabel('Expression Level')
            axes[0, 0].set_ylabel('Number of Transcripts')
            axes[0, 0].set_title('Transcript Expression Levels')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Isoform types
            isoform_types = [t['isoform_type'] for t in self.transcripts]
            isoform_counts = Counter(isoform_types)
            axes[0, 1].pie(isoform_counts.values(), labels=isoform_counts.keys(), autopct='%1.1f%%')
            axes[0, 1].set_title('Isoform Types')
            
            # Supporting reads
            supporting_reads = [t['supporting_reads'] for t in self.transcripts]
            axes[1, 0].hist(supporting_reads, bins=15, alpha=0.7, color='purple')
            axes[1, 0].set_xlabel('Supporting Reads')
            axes[1, 0].set_ylabel('Number of Transcripts')
            axes[1, 0].set_title('Read Support Distribution')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Average transcript lengths
            avg_lengths = [t['average_length'] for t in self.transcripts]
            axes[1, 1].hist(avg_lengths, bins=20, alpha=0.7, color='orange')
            axes[1, 1].set_xlabel('Average Length (bp)')
            axes[1, 1].set_ylabel('Number of Transcripts')
            axes[1, 1].set_title('Transcript Length Distribution')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            transcripts_plot = os.path.join(self.plots_dir, 'transcript_discovery.png')
            plt.savefig(transcripts_plot, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files.append(transcripts_plot)
        
        print(f"Generated {len(plot_files)} analysis plots")
        return plot_files
    
    def save_results(self):
        """Save all analysis results"""
        # Save HiFi reads
        reads_path = os.path.join(self.reads_dir, 'hifi_reads.json')
        with open(reads_path, 'w') as f:
            json.dump(self.hifi_reads, f, indent=2, default=str)
        
        # Save variants
        variants_path = os.path.join(self.variants_dir, 'detected_variants.json')
        with open(variants_path, 'w') as f:
            json.dump(self.variants, f, indent=2, default=str)
        
        # Save transcripts
        transcripts_path = os.path.join(self.transcripts_dir, 'discovered_transcripts.json')
        with open(transcripts_path, 'w') as f:
            json.dump(self.transcripts, f, indent=2, default=str)
        
        # Generate comprehensive report
        report_path = os.path.join(self.analysis_dir, 'pacbio_smrt_report.txt')
        self.generate_report(report_path)
        
        print(f"Results saved:")
        print(f"  - HiFi reads: {reads_path}")
        print(f"  - Variants: {variants_path}")
        print(f"  - Transcripts: {transcripts_path}")
        print(f"  - Report: {report_path}")
        
        return reads_path, variants_path, transcripts_path, report_path
    
    def generate_report(self, report_path):
        """Generate comprehensive analysis report"""
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("PacBio SMRT Sequencing Analysis Report\n")
            f.write("=" * 45 + "\n\n")
            
            f.write("OVERVIEW\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total HiFi reads: {len(self.hifi_reads)}\n")
            f.write(f"Total variants detected: {len(self.variants)}\n")
            f.write(f"Transcripts discovered: {len(self.transcripts)}\n")
            f.write(f"Analysis timestamp: {datetime.now().isoformat()}\n\n")
            
            if self.hifi_reads:
                # HiFi reads summary
                lengths = [r['length'] for r in self.hifi_reads]
                accuracies = [r['accuracy'] for r in self.hifi_reads]
                gc_contents = [r['gc_content'] for r in self.hifi_reads]
                
                f.write("HIFI READS ANALYSIS\n")
                f.write("-" * 25 + "\n")
                f.write(f"Average read length: {np.mean(lengths):.0f} bp\n")
                f.write(f"Read length range: {min(lengths):.0f} - {max(lengths):.0f} bp\n")
                f.write(f"Average accuracy: {np.mean(accuracies):.3f} ({np.mean(accuracies)*100:.1f}%)\n")
                f.write(f"Average GC content: {np.mean(gc_contents):.1f}%\n")
                
                # Target gene coverage
                genes = [r['target_gene'] for r in self.hifi_reads]
                gene_counts = Counter(genes)
                f.write(f"Genes covered: {len(gene_counts)}\n")
                f.write("Top genes by read count:\n")
                for gene, count in gene_counts.most_common(5):
                    f.write(f"  {gene}: {count} reads\n")
                f.write("\n")
            
            if self.variants:
                # Variants summary
                variant_types = Counter([v['type'] for v in self.variants])
                pathogenicity = Counter([v['pathogenicity'] for v in self.variants])
                
                f.write("VARIANT DETECTION\n")
                f.write("-" * 20 + "\n")
                f.write("Variant types:\n")
                for vtype, count in variant_types.items():
                    f.write(f"  {vtype}: {count}\n")
                
                f.write("Pathogenicity predictions:\n")
                for path, count in pathogenicity.items():
                    f.write(f"  {path}: {count}\n")
                
                # High-impact variants
                high_impact = [v for v in self.variants if v['pathogenicity'] == 'Likely Pathogenic']
                f.write(f"\nHigh-impact variants: {len(high_impact)}\n")
                if high_impact:
                    f.write("Top high-impact variants:\n")
                    for variant in high_impact[:5]:
                        f.write(f"  {variant['gene']}: {variant['type']} at position {variant['position']}\n")
                f.write("\n")
            
            if self.transcripts:
                # Transcript discovery summary
                isoform_types = Counter([t['isoform_type'] for t in self.transcripts])
                expression_levels = [t['expression_level'] for t in self.transcripts]
                
                f.write("TRANSCRIPT DISCOVERY\n")
                f.write("-" * 25 + "\n")
                f.write("Isoform types discovered:\n")
                for itype, count in isoform_types.items():
                    f.write(f"  {itype}: {count}\n")
                
                f.write(f"Average expression level: {np.mean(expression_levels):.1f}\n")
                f.write(f"Expression range: {min(expression_levels):.1f} - {max(expression_levels):.1f}\n")
                
                # Top expressed transcripts
                top_transcripts = sorted(self.transcripts, key=lambda x: x['expression_level'], reverse=True)[:5]
                f.write("Top expressed transcripts:\n")
                for transcript in top_transcripts:
                    f.write(f"  {transcript['transcript_id']} ({transcript['gene_name']}): {transcript['expression_level']:.1f}\n")
                f.write("\n")
            
            f.write("RECOMMENDATIONS FOR mRNA THERAPEUTICS\n")
            f.write("-" * 40 + "\n")
            f.write("• Use high-accuracy HiFi reads for rare variant detection\n")
            f.write("• Focus on high-impact variants in target genes for therapeutic design\n")
            f.write("• Consider transcript isoforms for comprehensive target coverage\n")
            f.write("• Validate pathogenic variants with additional sequencing methods\n")
            f.write("• Prioritize genes with multiple isoforms for mRNA target design\n")

def main():
    parser = argparse.ArgumentParser(description='PacBio SMRT Analysis Pipeline using RDKit')
    parser.add_argument('--create_sample', action='store_true', help='Create sample HiFi reads for testing')
    parser.add_argument('--input_reads', help='Input HiFi reads file (JSON format)')
    parser.add_argument('--output_dir', default='pacbio_smrt_output', help='Output directory')
    parser.add_argument('--num_reads', type=int, default=1000, help='Number of sample reads to generate')
    parser.add_argument('--target_genes', nargs='+', help='Target genes for analysis')
    parser.add_argument('--min_accuracy', type=float, default=0.99, help='Minimum HiFi read accuracy')
    parser.add_argument('--min_length', type=int, default=5000, help='Minimum read length')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = PacBioSMRTProcessor(output_dir=args.output_dir)
    
    # Load or create HiFi reads
    if args.create_sample:
        print("Creating sample HiFi reads for SMRT analysis...")
        processor.create_sample_hifi_reads(
            num_reads=args.num_reads,
            target_genes=args.target_genes
        )
    elif args.input_reads:
        print(f"Loading HiFi reads from {args.input_reads}")
        try:
            with open(args.input_reads, 'r') as f:
                processor.hifi_reads = json.load(f)
            print(f"Loaded {len(processor.hifi_reads)} HiFi reads")
        except Exception as e:
            print(f"Error loading reads: {e}")
            return
    else:
        print("Error: Specify input using:")
        print("  --create_sample (generate sample HiFi reads)")
        print("  --input_reads file.json (load existing reads)")
        return
    
    if not processor.hifi_reads:
        print("No HiFi reads available for analysis")
        return
    
    print(f"\nStarting PacBio SMRT analysis...")
    print(f"HiFi reads: {len(processor.hifi_reads)}")
    print(f"Minimum accuracy: {args.min_accuracy}")
    print(f"Minimum length: {args.min_length}")
    
    # Filter reads by quality thresholds
    original_count = len(processor.hifi_reads)
    processor.hifi_reads = [
        read for read in processor.hifi_reads 
        if read['accuracy'] >= args.min_accuracy and read['length'] >= args.min_length
    ]
    filtered_count = len(processor.hifi_reads)
    
    print(f"Filtered reads: {original_count} → {filtered_count}")
    
    # Perform variant detection
    print("\n" + "="*60)
    print("VARIANT DETECTION")
    print("="*60)
    variants = processor.detect_variants()
    
    # Perform transcript discovery
    print("\n" + "="*60)
    print("TRANSCRIPT DISCOVERY")
    print("="*60)
    transcripts = processor.discover_transcripts()
    
    # Generate analysis plots
    print("\n" + "="*60)
    print("GENERATING ANALYSIS PLOTS")
    print("="*60)
    plot_files = processor.generate_analysis_plots()
    
    # Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    results_paths = processor.save_results()
    
    # Print summary
    print(f"\nPacBio SMRT Analysis Complete!")
    print(f"Results saved in: {args.output_dir}")
    print(f"Generated plots: {len(plot_files)}")
    
    # Summary statistics
    if variants:
        high_quality_variants = [v for v in variants if v['quality_score'] > 40]
        pathogenic_variants = [v for v in variants if v['pathogenicity'] == 'Likely Pathogenic']
        
        print(f"\nVariant Summary:")
        print(f"  Total variants: {len(variants)}")
        print(f"  High-quality variants: {len(high_quality_variants)}")
        print(f"  Likely pathogenic: {len(pathogenic_variants)}")
        
        # Variant types
        variant_types = Counter([v['type'] for v in variants])
        for vtype, count in variant_types.items():
            print(f"  {vtype}: {count}")
    
    if transcripts:
        print(f"\nTranscript Summary:")
        print(f"  Total transcripts: {len(transcripts)}")
        
        # Genes with multiple isoforms
        gene_isoforms = defaultdict(int)
        for transcript in transcripts:
            gene_isoforms[transcript['gene_name']] += 1
        
        multi_isoform_genes = {gene: count for gene, count in gene_isoforms.items() if count > 1}
        print(f"  Genes with multiple isoforms: {len(multi_isoform_genes)}")
        
        if multi_isoform_genes:
            print("  Top genes by isoform count:")
            for gene, count in sorted(multi_isoform_genes.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"    {gene}: {count} isoforms")
    
    print(f"\nOutput files:")
    print(f"  - Analysis plots: {processor.plots_dir}")
    print(f"  - Detailed results: {processor.analysis_dir}")
    print(f"  - HiFi reads: {processor.reads_dir}")
    print(f"  - Variants: {processor.variants_dir}")
    print(f"  - Transcripts: {processor.transcripts_dir}")

if __name__ == "__main__":
    main()

# Example usage:
# Create sample HiFi reads and analyze:
# python smrt.py --create_sample

# Analyze with specific target genes:
# python smrt.py --create_sample --target_genes BRCA1 BRCA2 TP53

# Generate larger dataset:
# python smrt.py --create_sample --num_reads 2000

# Analyze with quality filters:
# python smrt.py --create_sample --min_accuracy 0.995 --min_length 8000

# Load existing HiFi reads:
# python smrt.py --input_reads hifi_reads.json

# Install requirements:
# pip install rdkit matplotlib seaborn pandas numpy