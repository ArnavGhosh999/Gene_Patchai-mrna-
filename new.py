#!/usr/bin/env python3
"""
Enhanced Bioinformatics Analysis Platform - Streamlit App (FIXED)
================================================================

Unified interface for multiple bioinformatics pipelines with file upload support,
automatic sample generation, comprehensive statistics, and interactive visualizations.

Requirements:
pip install streamlit pandas numpy matplotlib seaborn biopython plotly
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import tempfile
from pathlib import Path
import subprocess
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from collections import Counter, defaultdict

# Import the pipeline modules
try:
    import biosimai
    import fastqc
    import gene_optimizer
    import iedb
    import linear_design
    import mhcflurry_download
    import nanopore
    import sequencing
    import smrt
    import visium
    MODULES_AVAILABLE = True
except ImportError as e:
    st.error(f"Some modules not available: {e}")
    MODULES_AVAILABLE = False

# Configure Streamlit page
st.set_page_config(
    page_title="Enhanced Bioinformatics Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
    }
    .upload-section {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #4CAF50;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #e2f3ff;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .file-status {
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.25rem 0;
    }
    .file-uploaded {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    .file-missing {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

class EnhancedAnalysisRunner:
    """Enhanced analysis runner with file upload support and comprehensive results"""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.results = {}
        self.plots = {}
        self.uploaded_files = {}
        
    def save_uploaded_files(self, fasta_files, json_file, jsonl_file):
        """Save uploaded files to temporary directory"""
        saved_files = {}
        
        # Save FASTA files
        if fasta_files:
            fasta_paths = []
            for i, fasta_file in enumerate(fasta_files):
                fasta_path = os.path.join(self.temp_dir, f'reference_{i}.fasta')
                with open(fasta_path, 'wb') as f:
                    f.write(fasta_file.getbuffer())
                fasta_paths.append(fasta_path)
            saved_files['fasta_files'] = fasta_paths
            saved_files['primary_fasta'] = fasta_paths[0]  # Use first FASTA as primary
        
        # Save JSON file
        if json_file:
            json_path = os.path.join(self.temp_dir, 'dataset_catalog.json')
            with open(json_path, 'wb') as f:
                f.write(json_file.getbuffer())
            saved_files['json_file'] = json_path
        
        # Save JSONL file
        if jsonl_file:
            jsonl_path = os.path.join(self.temp_dir, 'assembly_report.jsonl')
            with open(jsonl_path, 'wb') as f:
                f.write(jsonl_file.getbuffer())
            saved_files['jsonl_file'] = jsonl_path
        
        self.uploaded_files = saved_files
        return saved_files
    
    def run_mrna_simulation(self, create_sample=True):
        """Run mRNA simulation with sample generation or uploaded files"""
        st.subheader("🧪 mRNA Simulation Platform")
        
        with st.spinner("Running mRNA simulation..."):
            try:
                # Initialize platform
                platform = biosimai.mRNASimulationPlatform(output_dir=os.path.join(self.temp_dir, 'mrna_sim'))
                
                if create_sample:
                    # Create sample sequences
                    sequences = platform.create_sample_mrna_sequences(num_sequences=3)
                else:
                    # Use uploaded FASTA file
                    sequences = platform.load_mrna_from_fasta(self.uploaded_files['primary_fasta'])
                
                all_results = []
                for seq_info in sequences[:3]:  # Limit to 3 for demo
                    result = platform.run_comprehensive_simulation(seq_info['sequence'], seq_info['id'])
                    if result:
                        all_results.append(result)
                
                # Calculate summary statistics with realistic values
                stats = {
                    "Total Sequences": len(sequences),
                    "Successful Simulations": len(all_results),
                    "Average Stability": f"{np.mean([r['properties']['predicted_stability'] for r in all_results]):.1f}",
                    "Average Translation Efficiency": f"{np.mean([r['properties']['translation_efficiency'] for r in all_results]):.1f}",
                    "Average GC Content": f"{np.mean([r['properties']['gc_content'] for r in all_results]):.1f}%",
                    "Average Molecular Weight": f"{np.mean([r['properties']['molecular_weight'] for r in all_results])/1000:.0f} kDa"
                }
                
                # Create visualization
                fig = self.create_mrna_simulation_plots(all_results)
                
                self.results['mRNA Simulation'] = {
                    'stats': stats,
                    'detailed_results': all_results,
                    'status': 'success'
                }
                self.plots['mRNA Simulation'] = fig
                
                return stats, fig
                
            except Exception as e:
                st.error(f"mRNA simulation failed: {e}")
                return None, None
    
    def run_fastqc_analysis(self, create_sample=True):
        """Run FastQC analysis with sample generation"""
        st.subheader("📊 FastQC Quality Control")
        
        with st.spinner("Running FastQC analysis..."):
            try:
                # Initialize processor
                processor = fastqc.FastQCProcessor(output_dir=os.path.join(self.temp_dir, 'fastqc'))
                
                if create_sample:
                    # Create sample FASTQ files
                    sample_files = []
                    for i in range(3):
                        sample_file = os.path.join(self.temp_dir, f'sample_{i}.fastq')
                        processor.create_sample_fastq(sample_file, num_reads=np.random.randint(5000, 15000))
                        sample_files.append(sample_file)
                    
                    # Process files
                    processed_samples = 0
                    all_results = []
                    for fastq_file in sample_files:
                        result = processor.process_sample(fastq_file)
                        if result:
                            processed_samples += 1
                            all_results.append(result)
                    
                    # Generate batch report
                    if processed_samples > 0:
                        processor.generate_batch_report()
                    
                    # Calculate summary statistics
                    stats = {
                        "FASTQ Files Processed": processed_samples,
                        "Total Reads": sum(r['basic_statistics']['total_sequences'] for r in all_results),
                        "Average Quality Score": f"{np.mean([r['quality_analysis']['mean_quality_score'] for r in all_results]):.1f}",
                        "Average GC Content": f"{np.mean([r['gc_analysis']['mean_gc_content'] for r in all_results]):.1f}%",
                        "Q30+ Bases": f"{np.mean([r['quality_analysis']['q30_bases_percent'] for r in all_results]):.1f}%",
                        "Adapter Contamination": f"{np.mean([r['adapter_contamination']['contamination_rate_percent'] for r in all_results]):.1f}%"
                    }
                    
                    # Create visualization
                    fig = self.create_fastqc_plots(all_results)
                    
                    self.results['FastQC'] = {
                        'stats': stats,
                        'detailed_results': all_results,
                        'status': 'success'
                    }
                    self.plots['FastQC'] = fig
                    
                    return stats, fig
                
            except Exception as e:
                st.error(f"FastQC analysis failed: {e}")
                return None, None
    
    def run_gene_optimization(self, create_sample=True):
        """Run gene optimization"""
        st.subheader("🔧 Gene Sequence Optimization")
        
        with st.spinner("Running gene optimization..."):
            try:
                # Initialize optimizer
                optimizer = gene_optimizer.GeneOptimizer(
                    reference_fasta=self.uploaded_files['primary_fasta'],
                    dataset_json=self.uploaded_files['json_file'],
                    assembly_jsonl=self.uploaded_files['jsonl_file'],
                    output_dir=os.path.join(self.temp_dir, 'gene_opt')
                )
                
                if create_sample:
                    # Create sample sequences
                    sequences = optimizer.create_sample_sequences(num_sequences=5)
                else:
                    # Extract from reference
                    sequences = optimizer.extract_cds_from_fasta(max_sequences=5)
                
                # Run optimization
                results = optimizer.batch_optimize_sequences(sequences, target_gc=50)
                
                # Calculate summary statistics
                successful_results = [r for r in results if r.get('optimization_successful', False)]
                
                stats = {
                    "Total Sequences": len(sequences),
                    "Successful Optimizations": len(successful_results),
                    "Success Rate": f"{(len(successful_results) / len(sequences)) * 100:.1f}%",
                    "Average GC Change": f"{np.mean([r['improvements']['gc_content_change'] for r in successful_results]):.1f}%",
                    "Average CAI Improvement": f"{np.mean([r['improvements']['cai_improvement'] for r in successful_results]):.3f}",
                    "Rare Codons Reduced": f"{np.mean([r['improvements']['rare_codons_reduced'] for r in successful_results]):.1f}"
                }
                
                # Create visualization
                fig = self.create_gene_optimization_plots(successful_results)
                
                self.results['Gene Optimization'] = {
                    'stats': stats,
                    'detailed_results': successful_results,
                    'status': 'success'
                }
                self.plots['Gene Optimization'] = fig
                
                return stats, fig
                
            except Exception as e:
                st.error(f"Gene optimization failed: {e}")
                return None, None
    
    def run_vaccine_design(self):
        """Run mRNA vaccine design - FIXED VERSION"""
        st.subheader("💉 mRNA Vaccine Design")
        
        with st.spinner("Running vaccine design..."):
            try:
                # Initialize designer
                designer = iedb.AdvancedmRNAVaccineDesigner(
                    self.uploaded_files['primary_fasta'], 
                    self.uploaded_files['json_file'], 
                    self.uploaded_files['jsonl_file']
                )
                
                # Run comprehensive pipeline
                results = designer.run_comprehensive_pipeline()
                
                if 'error' not in results:
                    stats = {
                        "Sequences Processed": results['summary']['sequences_processed'],
                        "Peptides Generated": results['summary']['peptides_generated'],
                        "MHC Predictions": results['summary']['mhc_predictions'],
                        "B-cell Epitopes": results['summary']['bcell_epitopes'],
                        "Construct Epitopes": results['summary']['construct_epitopes'],
                        "Global Coverage": f"{results['population_coverage'].get('global', {}).get('weighted_average_coverage', 0):.1%}"
                    }
                    
                    # Create visualization - FIXED
                    fig = self.create_vaccine_design_plots_fixed(results)
                    
                    self.results['Vaccine Design'] = {
                        'stats': stats,
                        'detailed_results': results,
                        'status': 'success'
                    }
                    self.plots['Vaccine Design'] = fig
                    
                    return stats, fig
                else:
                    st.error(f"Vaccine design failed: {results['error']}")
                    return None, None
                
            except Exception as e:
                st.error(f"Vaccine design failed: {e}")
                return None, None
    
    def run_mrna_folding(self, create_sample=True):
        """Run mRNA folding optimization"""
        st.subheader("🔀 mRNA Folding Optimization")
        
        with st.spinner("Running mRNA folding optimization..."):
            try:
                # Initialize optimizer
                optimizer = linear_design.mRNAFoldingOptimizer()
                
                # Example protein sequence
                example_protein = "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT"
                
                # Run design
                design_result = optimizer.design_complete_mrna(
                    example_protein[:200],  # Use shorter sequence for demo
                    include_utrs=True,
                    kozak_optimization=True
                )
                
                if 'error' not in design_result:
                    summary = design_result['design_summary']
                    
                    stats = {
                        "mRNA Length": f"{summary['total_length']} nt",
                        "GC Content": summary['gc_content'],
                        "Translation Efficiency": summary['predicted_translation_efficiency'],
                        "mRNA Half-life": summary['predicted_mrna_half_life'],
                        "Structural Stability": summary['structural_stability'],
                        "Manufacturing Feasibility": summary['manufacturing_feasibility']
                    }
                    
                    # Create visualization
                    fig = self.create_folding_plots(design_result)
                    
                    self.results['mRNA Folding'] = {
                        'stats': stats,
                        'detailed_results': design_result,
                        'status': 'success'
                    }
                    self.plots['mRNA Folding'] = fig
                    
                    return stats, fig
                else:
                    st.error(f"mRNA folding failed: {design_result['error']}")
                    return None, None
                
            except Exception as e:
                st.error(f"mRNA folding failed: {e}")
                return None, None
    
    def run_mhcflurry_analysis(self):
        """Run MHCflurry analysis"""
        st.subheader("🎯 MHCflurry Integration")
        
        with st.spinner("Running MHCflurry analysis..."):
            try:
                # Initialize predictor
                predictor = mhcflurry_download.IEDBmRNAPredictor(
                    self.uploaded_files['primary_fasta'],
                    self.uploaded_files['json_file'],
                    self.uploaded_files['jsonl_file']
                )
                
                # Run epitope pipeline
                results = predictor.run_epitope_pipeline()
                
                if 'error' not in results:
                    stats = {
                        "Sequences Processed": results['summary']['total_sequences'],
                        "Peptides Generated": results['summary']['total_peptides'],
                        "IEDB Predictions": results['summary']['iedb_predictions'],
                        "Simple Scores": results['summary']['simple_scores'],
                        "MHCflurry Predictions": results['summary']['mhcflurry_predictions']
                    }
                    
                    # Create visualization
                    fig = self.create_mhcflurry_plots(results)
                    
                    self.results['MHCflurry'] = {
                        'stats': stats,
                        'detailed_results': results,
                        'status': 'success'
                    }
                    self.plots['MHCflurry'] = fig
                    
                    return stats, fig
                else:
                    st.error(f"MHCflurry analysis failed: {results['error']}")
                    return None, None
                
            except Exception as e:
                st.error(f"MHCflurry analysis failed: {e}")
                return None, None
    
    def run_nanopore_analysis(self, create_sample=True):
        """Run Nanopore analysis"""
        st.subheader("🧬 Nanopore Sequencing Analysis")
        
        with st.spinner("Running Nanopore analysis..."):
            try:
                # Initialize processor
                processor = nanopore.NanoporeProcessor(
                    self.uploaded_files['primary_fasta'],
                    self.uploaded_files['json_file'],
                    self.uploaded_files['jsonl_file']
                )
                
                if create_sample:
                    # Create sample FAST5 files
                    sample_files = []
                    for i in range(3):
                        sample_file = os.path.join(self.temp_dir, f'sample_{i}.fast5')
                        nanopore.create_sample_fast5(sample_file, num_reads=5)
                        sample_files.append(sample_file)
                    
                    # Process files
                    all_processed_reads = []
                    for fast5_file in sample_files:
                        reads_data = processor.read_fast5_file(fast5_file)
                        if reads_data:
                            processed_reads = processor.process_reads(reads_data)
                            all_processed_reads.extend(processed_reads)
                    
                    if all_processed_reads:
                        stats = {
                            "FAST5 Files Processed": len(sample_files),
                            "Total Reads": len(all_processed_reads),
                            "Average Read Length": f"{np.mean([r['length'] for r in all_processed_reads]):.0f} bp",
                            "Average Quality Score": f"{np.mean([r['quality_score'] for r in all_processed_reads]):.1f}",
                            "Total Bases Called": sum(r['length'] for r in all_processed_reads),
                            "Mean Signal Intensity": f"{np.mean([r['mean_signal'] for r in all_processed_reads]):.1f}"
                        }
                        
                        # Create visualization
                        fig = self.create_nanopore_plots(all_processed_reads)
                        
                        self.results['Nanopore'] = {
                            'stats': stats,
                            'detailed_results': all_processed_reads,
                            'status': 'success'
                        }
                        self.plots['Nanopore'] = fig
                        
                        return stats, fig
                    else:
                        st.error("No reads were successfully processed")
                        return None, None
                
            except Exception as e:
                st.error(f"Nanopore analysis failed: {e}")
                return None, None
    
    def run_illumina_analysis(self, create_sample=True):
        """Run Illumina analysis"""
        st.subheader("🔍 Illumina Sequencing Analysis")
        
        with st.spinner("Running Illumina analysis..."):
            try:
                # Initialize processor
                processor = sequencing.IlluminaProcessor(reference_fasta=self.uploaded_files['primary_fasta'])
                
                # Load metrics
                processor.load_interop_metrics()
                
                if create_sample:
                    # Create sample VCF
                    vcf_file = os.path.join(self.temp_dir, 'sample_variants.vcf')
                    processor.create_sample_vcf(vcf_file, num_variants=1000)
                    
                    # Load and analyze VCF data
                    variants = processor.load_vcf_data(vcf_file)
                    
                    if variants:
                        # Perform analyses
                        snp_analysis = processor.detect_snps()
                        indel_analysis = processor.detect_indels()
                        af_analysis = processor.analyze_allele_frequencies()
                        disease_variants = processor.identify_disease_linked_mutations()
                        
                        stats = {
                            "Total Variants": len(processor.variants['variants/POS']),
                            "SNPs Detected": snp_analysis['total_snps'] if snp_analysis else 0,
                            "Indels Detected": indel_analysis['total_indels'] if indel_analysis else 0,
                            "High Quality SNPs": snp_analysis['high_quality_snps'] if snp_analysis else 0,
                            "Disease Gene Variants": len(disease_variants) if disease_variants else 0,
                            "Mean Variant Quality": f"{np.mean(processor.variants['variants/QUAL']):.1f}"
                        }
                        
                        # Create visualization
                        fig = self.create_illumina_plots(snp_analysis, indel_analysis, af_analysis)
                        
                        self.results['Illumina'] = {
                            'stats': stats,
                            'detailed_results': {
                                'snps': snp_analysis,
                                'indels': indel_analysis,
                                'allele_freq': af_analysis,
                                'disease': disease_variants
                            },
                            'status': 'success'
                        }
                        self.plots['Illumina'] = fig
                        
                        return stats, fig
                    else:
                        st.error("Failed to load VCF data")
                        return None, None
                
            except Exception as e:
                st.error(f"Illumina analysis failed: {e}")
                return None, None
    
    def run_pacbio_analysis(self, create_sample=True):
        """Run PacBio analysis"""
        st.subheader("⚗️ PacBio SMRT Analysis")
        
        with st.spinner("Running PacBio analysis..."):
            try:
                # Initialize processor
                processor = smrt.PacBioSMRTProcessor(output_dir=os.path.join(self.temp_dir, 'pacbio'))
                
                if create_sample:
                    # Create sample HiFi reads
                    processor.create_sample_hifi_reads(num_reads=500)
                    
                    # Filter reads by quality
                    original_count = len(processor.hifi_reads)
                    processor.hifi_reads = [
                        read for read in processor.hifi_reads 
                        if read['accuracy'] >= 0.99 and read['length'] >= 5000
                    ]
                    filtered_count = len(processor.hifi_reads)
                    
                    # Perform analyses
                    variants = processor.detect_variants()
                    transcripts = processor.discover_transcripts()
                    
                    stats = {
                        "HiFi Reads Generated": f"{filtered_count} (filtered from {original_count})",
                        "Average Read Length": f"{np.mean([r['length'] for r in processor.hifi_reads]):.0f} bp",
                        "Average Accuracy": f"{np.mean([r['accuracy'] for r in processor.hifi_reads]):.3f}",
                        "Variants Detected": len(variants),
                        "Transcripts Discovered": len(transcripts),
                        "Pathogenic Variants": len([v for v in variants if v.get('pathogenicity') == 'Likely Pathogenic'])
                    }
                    
                    # Create visualization
                    fig = self.create_pacbio_plots(processor.hifi_reads, variants, transcripts)
                    
                    self.results['PacBio'] = {
                        'stats': stats,
                        'detailed_results': {
                            'reads': processor.hifi_reads,
                            'variants': variants,
                            'transcripts': transcripts
                        },
                        'status': 'success'
                    }
                    self.plots['PacBio'] = fig
                    
                    return stats, fig
                
            except Exception as e:
                st.error(f"PacBio analysis failed: {e}")
                return None, None
    
    def run_spatial_analysis(self, create_sample=True):
        """Run spatial transcriptomics analysis - FIXED VERSION"""
        st.subheader("🗺️ Spatial Transcriptomics")
        
        with st.spinner("Running spatial analysis..."):
            try:
                # Initialize processor
                processor = visium.SpatialTranscriptomicsProcessor(output_dir=os.path.join(self.temp_dir, 'spatial'))
                
                if create_sample:
                    # Create sample spatial data
                    adata = processor.create_sample_spatial_data(n_spots=2000, n_genes=500)
                    
                    if adata is not None:
                        # Store original data
                        processor.adata = adata
                        
                        # FIXED: Initialize processor.results with basic summary
                        processor.results = {
                            'summary': {
                                'n_spots': int(adata.n_obs),
                                'n_genes': int(adata.n_vars),
                                'total_umi_counts': int(adata.obs['total_counts'].sum()),
                                'mean_genes_per_spot': float(adata.obs['n_genes'].mean()),
                                'mean_umi_per_spot': float(adata.obs['total_counts'].mean()),
                                'processing_timestamp': datetime.now().isoformat()
                            }
                        }
                        
                        # Preprocessing
                        adata_processed = processor.preprocess_data(adata)
                        
                        # Update summary after preprocessing
                        if adata_processed is not None:
                            processor.results['summary'].update({
                                'n_spots_after_filter': int(adata_processed.n_obs),
                                'n_genes_after_filter': int(adata_processed.n_vars)
                            })
                        
                        # Analyses
                        adata_processed = processor.spatial_network_analysis(adata_processed)
                        adata_processed = processor.dimensionality_reduction(adata_processed)
                        adata_processed = processor.spatial_clustering(adata_processed)
                        adata_processed = processor.gene_expression_patterns(adata_processed)
                        
                        # mRNA localization analysis
                        localization_results = processor.mrna_localization_analysis(adata_processed)
                        
                        # Ensure we have all required data for stats
                        stats = {
                            "Spots Analyzed": processor.results['summary']['n_spots'],
                            "Genes Analyzed": processor.results['summary']['n_genes'],
                            "Spatial Clusters": processor.results.get('n_clusters', 'N/A'),
                            "Mean UMI per Spot": f"{processor.results['summary']['mean_umi_per_spot']:.1f}",
                            "Mean Genes per Spot": f"{processor.results['summary']['mean_genes_per_spot']:.1f}",
                            "High Priority Targets": len([g for g, data in localization_results.items() 
                                                        if data.get('targeting_priority') == 'High'])
                        }
                        
                        # Create visualization
                        fig = self.create_spatial_plots(adata_processed, localization_results)
                        
                        self.results['Spatial'] = {
                            'stats': stats,
                            'detailed_results': {
                                'adata': adata_processed,
                                'localization': localization_results,
                                'processor_results': processor.results
                            },
                            'status': 'success'
                        }
                        self.plots['Spatial'] = fig
                        
                        return stats, fig
                    else:
                        st.error("Failed to create spatial data")
                        return None, None
                
            except Exception as e:
                st.error(f"Spatial analysis failed: {e}")
                import traceback
                traceback.print_exc()
                return None, None
    
    # Visualization creation methods
    def create_mrna_simulation_plots(self, results):
        """Create mRNA simulation visualization plots"""
        if not results:
            return None
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Stability Scores', 'Translation Efficiency', 'GC Content Distribution', 'Molecular Weight'),
        )
        
        # Stability scores
        stabilities = [r['properties']['predicted_stability'] for r in results]
        fig.add_trace(
            go.Bar(x=[r['sequence_id'] for r in results], y=stabilities, name='Stability', marker_color='lightblue'),
            row=1, col=1
        )
        
        # Translation efficiency
        efficiencies = [r['properties']['translation_efficiency'] for r in results]
        fig.add_trace(
            go.Bar(x=[r['sequence_id'] for r in results], y=efficiencies, name='Translation Efficiency', marker_color='lightgreen'),
            row=1, col=2
        )
        
        # GC content distribution
        gc_contents = [r['properties']['gc_content'] for r in results]
        fig.add_trace(
            go.Histogram(x=gc_contents, name='GC Content', marker_color='orange'),
            row=2, col=1
        )
        
        # Molecular weight
        mol_weights = [r['properties']['molecular_weight'] for r in results]
        fig.add_trace(
            go.Bar(x=[r['sequence_id'] for r in results], y=mol_weights, name='Molecular Weight', marker_color='purple'),
            row=2, col=2
        )
        
        fig.update_layout(height=600, title_text="mRNA Simulation Results", showlegend=False)
        return fig
    
    def create_fastqc_plots(self, results):
        """Create FastQC visualization plots"""
        if not results:
            return None
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Quality Scores', 'GC Content', 'Sequence Counts', 'Adapter Contamination'),
        )
        
        # Quality scores
        quality_scores = [r['quality_analysis']['mean_quality_score'] for r in results]
        sample_names = [r['sample_name'] for r in results]
        fig.add_trace(
            go.Bar(x=sample_names, y=quality_scores, name='Quality Score'),
            row=1, col=1
        )
        
        # GC content
        gc_contents = [r['gc_analysis']['mean_gc_content'] for r in results]
        fig.add_trace(
            go.Bar(x=sample_names, y=gc_contents, name='GC Content'),
            row=1, col=2
        )
        
        # Sequence counts
        seq_counts = [r['basic_statistics']['total_sequences'] for r in results]
        fig.add_trace(
            go.Bar(x=sample_names, y=seq_counts, name='Total Sequences'),
            row=2, col=1
        )
        
        # Adapter contamination
        contamination = [r['adapter_contamination']['contamination_rate_percent'] for r in results]
        fig.add_trace(
            go.Bar(x=sample_names, y=contamination, name='Contamination %'),
            row=2, col=2
        )
        
        fig.update_layout(height=600, title_text="FastQC Analysis Results")
        return fig
    
    def create_gene_optimization_plots(self, results):
        """Create gene optimization visualization plots"""
        if not results:
            return None
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('GC Content Changes', 'CAI Improvements', 'GC Content Before vs After', 'Rare Codons Reduced'),
        )
        
        # GC content changes
        gc_changes = [r['improvements']['gc_content_change'] for r in results]
        seq_ids = [r['sequence_id'] for r in results]
        fig.add_trace(
            go.Bar(x=seq_ids, y=gc_changes, name='GC Change', marker_color='lightcoral'),
            row=1, col=1
        )
        
        # CAI improvements
        cai_improvements = [r['improvements']['cai_improvement'] for r in results]
        fig.add_trace(
            go.Bar(x=seq_ids, y=cai_improvements, name='CAI Improvement', marker_color='lightseagreen'),
            row=1, col=2
        )
        
        # Before/after GC content scatter plot
        original_gc = [r['original_properties']['gc_content'] for r in results]
        optimized_gc = [r['optimized_properties']['gc_content'] for r in results]
        fig.add_trace(
            go.Scatter(x=original_gc, y=optimized_gc, mode='markers', name='GC Optimization', 
                      marker=dict(size=10, color='blue')),
            row=2, col=1
        )
        # Add diagonal line for reference
        fig.add_trace(
            go.Scatter(x=[30, 70], y=[30, 70], mode='lines', name='No Change', 
                      line=dict(dash='dash', color='red')),
            row=2, col=1
        )
        
        # Rare codons reduced
        rare_codons = [r['improvements']['rare_codons_reduced'] for r in results]
        fig.add_trace(
            go.Bar(x=seq_ids, y=rare_codons, name='Rare Codons Reduced', marker_color='gold'),
            row=2, col=2
        )
        
        fig.update_layout(height=600, title_text="Gene Optimization Results", showlegend=False)
        return fig
    
    def create_vaccine_design_plots_fixed(self, results):
        """Create vaccine design visualization plots - FIXED VERSION"""
        if not results or 'error' in results:
            return None
        
        # Use regular subplots with all xy type
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Population Coverage', 'Epitope Types', 'MHC Binding', 'Construct Overview'),
            specs=[[{}, {}], [{}, {}]]  # All regular xy plots
        )
        
        # Population coverage
        coverage_data = results.get('population_coverage', {})
        populations = []
        coverages = []
        for pop, data in coverage_data.items():
            if pop != 'global' and isinstance(data, dict):
                populations.append(pop)
                coverages.append(data.get('average_coverage', 0) * 100)
        
        if populations:
            fig.add_trace(
                go.Bar(x=populations, y=coverages, name='Coverage %', marker_color='lightblue'),
                row=1, col=1
            )
        
        # Epitope types in construct - AS BAR CHART instead of pie
        construct = results.get('vaccine_construct', {})
        if construct:
            epitope_types = ['T-cell', 'B-cell']
            epitope_counts = [construct.get('tcell_epitopes', 0), construct.get('bcell_epitopes', 0)]
            fig.add_trace(
                go.Bar(x=epitope_types, y=epitope_counts, name='Epitope Types', marker_color='lightgreen'),
                row=1, col=2
            )
        
        # MHC predictions summary
        mhc_data = results.get('mhc_predictions', [])
        if mhc_data:
            binding_strengths = [pred.get('binding_strength', 'Unknown') for pred in mhc_data]
            strength_counts = Counter(binding_strengths)
            fig.add_trace(
                go.Bar(x=list(strength_counts.keys()), y=list(strength_counts.values()), 
                      name='Binding Strength', marker_color='lightcoral'),
                row=2, col=1
            )
        
        # Summary metrics
        summary = results.get('summary', {})
        metrics = ['Sequences', 'Peptides', 'MHC Predictions', 'Epitopes']
        values = [
            summary.get('sequences_processed', 0),
            summary.get('peptides_generated', 0),
            summary.get('mhc_predictions', 0),
            summary.get('construct_epitopes', 0)
        ]
        fig.add_trace(
            go.Bar(x=metrics, y=values, name='Pipeline Metrics', marker_color='gold'),
            row=2, col=2
        )
        
        fig.update_layout(height=600, title_text="mRNA Vaccine Design Results", showlegend=False)
        return fig
    
    def create_folding_plots(self, results):
        """Create mRNA folding visualization plots"""
        if not results or 'error' in results:
            return None
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Optimization Metrics', 'Structure Analysis', 'Manufacturing Score', 'Quality Profile'),
            specs=[[{}, {}], [{}, {"type": "polar"}]]
        )
        
        analysis = results['final_analysis']
        
        # Optimization metrics
        metrics = ['Translation Efficiency', 'Structural Stability', 'Manufacturing Feasibility']
        scores = [
            analysis['translation_efficiency'],
            analysis['structure_analysis']['stability_score'],
            analysis['manufacturing_feasibility']['feasibility_score']
        ]
        fig.add_trace(
            go.Bar(x=metrics, y=scores, name='Scores'),
            row=1, col=1
        )
        
        # Structure analysis
        struct_features = ['Hairpins', 'Repeats', 'Motifs']
        struct_counts = [
            len(analysis['structure_analysis']['hairpin_regions']),
            len(analysis['structure_analysis']['repeat_regions']),
            len(analysis['structure_analysis']['problematic_motifs'])
        ]
        fig.add_trace(
            go.Bar(x=struct_features, y=struct_counts, name='Structure Features'),
            row=1, col=2
        )
        
        # Manufacturing feasibility as gauge
        manufacturing_score = analysis['manufacturing_feasibility']['feasibility_score']
        fig.add_trace(
            go.Bar(x=['Manufacturing Score'], y=[manufacturing_score], name='Manufacturing'),
            row=2, col=1
        )
        
        # Overall quality assessment as radar
        quality_metrics = ['GC Content', 'Translation Eff', 'Stability', 'Immunogenicity']
        quality_scores = [
            analysis['gc_content'] * 100,
            analysis['translation_efficiency'],
            analysis['structure_analysis']['stability_score'],
            10 - analysis['immunogenicity_risk']['risk_score']
        ]
        fig.add_trace(
            go.Scatterpolar(r=quality_scores, theta=quality_metrics, fill='toself', name='Quality Profile'),
            row=2, col=2
        )
        
        fig.update_layout(height=600, title_text="mRNA Folding Optimization Results")
        return fig
    
    def create_mhcflurry_plots(self, results):
        """Create MHCflurry visualization plots"""
        if not results or 'error' in results:
            return None
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Analysis Summary', 'Peptide Scores', 'Length Distribution', 'Analysis Methods'),
        )
        
        summary = results.get('summary', {})
        
        # Analysis summary
        metrics = ['Total Sequences', 'Total Peptides', 'IEDB Predictions', 'Simple Scores']
        values = [
            summary.get('total_sequences', 0),
            summary.get('total_peptides', 0),
            summary.get('iedb_predictions', 0),
            summary.get('simple_scores', 0)
        ]
        fig.add_trace(
            go.Bar(x=metrics, y=values, name='Analysis Metrics', marker_color='lightblue'),
            row=1, col=1
        )
        
        # Peptide analysis
        peptide_data = results.get('peptide_analysis', {})
        if 'top_peptides_simple' in peptide_data:
            top_peptides = peptide_data['top_peptides_simple'][:10]
            peptides = [p['peptide'] for p in top_peptides]
            scores = [p['immunogenicity_score'] for p in top_peptides]
            
            fig.add_trace(
                go.Bar(x=peptides, y=scores, name='Immunogenicity Score', marker_color='lightcoral'),
                row=1, col=2
            )
        
        # Length distribution
        if 'length_distribution' in peptide_data:
            lengths = list(peptide_data['length_distribution'].keys())
            counts = list(peptide_data['length_distribution'].values())
            fig.add_trace(
                go.Bar(x=lengths, y=counts, name='Peptide Lengths', marker_color='lightgreen'),
                row=2, col=1
            )
        
        # Analysis methods as bar chart instead of pie
        analysis_types = ['IEDB', 'Simple Scoring', 'MHCflurry']
        type_counts = [
            summary.get('iedb_predictions', 0),
            summary.get('simple_scores', 0),
            summary.get('mhcflurry_predictions', 0)
        ]
        fig.add_trace(
            go.Bar(x=analysis_types, y=type_counts, name='Analysis Methods', marker_color='gold'),
            row=2, col=2
        )
        
        fig.update_layout(height=600, title_text="MHCflurry Analysis Results", showlegend=False)
        return fig
    
    def create_nanopore_plots(self, reads):
        """Create Nanopore analysis visualization plots"""
        if not reads:
            return None
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Read Length Distribution', 'Quality Scores', 'Signal Analysis', 'Channel Distribution'),
        )
        
        # Read length distribution
        lengths = [r['length'] for r in reads]
        fig.add_trace(
            go.Histogram(x=lengths, name='Read Lengths'),
            row=1, col=1
        )
        
        # Quality scores
        qualities = [r['quality_score'] for r in reads]
        fig.add_trace(
            go.Histogram(x=qualities, name='Quality Scores'),
            row=1, col=2
        )
        
        # Signal analysis
        mean_signals = [r['mean_signal'] for r in reads]
        fig.add_trace(
            go.Histogram(x=mean_signals, name='Mean Signal'),
            row=2, col=1
        )
        
        # Channel distribution
        channels = [r['channel'] for r in reads]
        channel_counts = Counter(channels)
        fig.add_trace(
            go.Bar(x=list(channel_counts.keys()), y=list(channel_counts.values()), name='Channel Usage'),
            row=2, col=2
        )
        
        fig.update_layout(height=600, title_text="Nanopore Sequencing Analysis")
        return fig
    
    def create_illumina_plots(self, snp_analysis, indel_analysis, af_analysis):
        """Create Illumina analysis visualization plots"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Variant Type Counts', 'SNP Quality Distribution', 'Allele Frequencies', 'Chromosome Distribution'),
        )
        
        # Variant types as bar chart instead of pie
        if snp_analysis and indel_analysis:
            variant_types = ['SNPs', 'Indels']
            variant_counts = [snp_analysis['total_snps'], indel_analysis['total_indels']]
            fig.add_trace(
                go.Bar(x=variant_types, y=variant_counts, name='Variant Types', marker_color=['lightblue', 'lightcoral']),
                row=1, col=1
            )
        
        # SNP chromosome distribution
        if snp_analysis and 'chr_counts' in snp_analysis:
            chromosomes = list(snp_analysis['chr_counts'].keys())
            counts = list(snp_analysis['chr_counts'].values())
            fig.add_trace(
                go.Bar(x=chromosomes, y=counts, name='SNPs per Chromosome', marker_color='lightgreen'),
                row=1, col=2
            )
        
        # Allele frequency distribution
        if af_analysis and 'af_distribution' in af_analysis:
            af_dist = af_analysis['af_distribution']
            fig.add_trace(
                go.Histogram(x=af_dist, name='Allele Frequencies', marker_color='orange'),
                row=2, col=1
            )
        
        # Quality metrics
        if snp_analysis and indel_analysis:
            metrics = ['Total SNPs', 'High Quality SNPs', 'Total Indels']
            values = [
                snp_analysis['total_snps'],
                snp_analysis['high_quality_snps'],
                indel_analysis['total_indels']
            ]
            fig.add_trace(
                go.Bar(x=metrics, y=values, name='Quality Metrics', marker_color='purple'),
                row=2, col=2
            )
        
        fig.update_layout(height=600, title_text="Illumina Sequencing Analysis", showlegend=False)
        return fig
    
    def create_pacbio_plots(self, reads, variants, transcripts):
        """Create PacBio analysis visualization plots"""
        if not reads:
            return None
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Read Accuracy', 'Read Lengths', 'Variant Types', 'Transcript Expression'),
        )
        
        # Read accuracy
        accuracies = [r['accuracy'] * 100 for r in reads]
        fig.add_trace(
            go.Histogram(x=accuracies, name='Read Accuracy %', marker_color='lightblue'),
            row=1, col=1
        )
        
        # Read lengths
        lengths = [r['length'] for r in reads]
        fig.add_trace(
            go.Histogram(x=lengths, name='Read Lengths', marker_color='lightgreen'),
            row=1, col=2
        )
        
        # Variant types as bar chart instead of pie
        if variants:
            variant_types = [v['type'] for v in variants]
            type_counts = Counter(variant_types)
            fig.add_trace(
                go.Bar(x=list(type_counts.keys()), y=list(type_counts.values()), 
                      name='Variant Types', marker_color='lightcoral'),
                row=2, col=1
            )
        
        # Transcript expression
        if transcripts:
            expr_levels = [t['expression_level'] for t in transcripts]
            fig.add_trace(
                go.Histogram(x=expr_levels, name='Expression Levels', marker_color='gold'),
                row=2, col=2
            )
        
        fig.update_layout(height=600, title_text="PacBio SMRT Analysis", showlegend=False)
        return fig
    
    def create_spatial_plots(self, adata, localization_results):
        """Create spatial transcriptomics visualization plots"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Spatial Overview', 'Gene Expression', 'Cluster Distribution', 'Target Priority'),
        )
        
        # Spatial coordinates overview
        if 'spatial' in adata.obsm:
            spatial_coords = adata.obsm['spatial']
            fig.add_trace(
                go.Scatter(x=spatial_coords[:, 0], y=spatial_coords[:, 1], 
                          mode='markers', name='Spots', marker=dict(size=3, color='lightblue')),
                row=1, col=1
            )
        
        # Gene expression distribution
        if adata.n_vars > 0:
            gene_expr = adata.X[:, 0] if hasattr(adata.X, '__getitem__') else adata.X.toarray()[:, 0]
            fig.add_trace(
                go.Histogram(x=gene_expr, name='Gene Expression', marker_color='lightgreen'),
                row=1, col=2
            )
        
        # Clustering results as bar chart instead of pie
        if 'leiden' in adata.obs.columns:
            cluster_counts = Counter(adata.obs['leiden'])
            fig.add_trace(
                go.Bar(x=list(cluster_counts.keys()), y=list(cluster_counts.values()), 
                      name='Clusters', marker_color='lightcoral'),
                row=2, col=1
            )
        
        # Target priority
        if localization_results:
            priorities = [data.get('targeting_priority', 'Unknown') for data in localization_results.values()]
            priority_counts = Counter(priorities)
            fig.add_trace(
                go.Bar(x=list(priority_counts.keys()), y=list(priority_counts.values()), 
                      name='Target Priority', marker_color='gold'),
                row=2, col=2
            )
        
        fig.update_layout(height=600, title_text="Spatial Transcriptomics Analysis", showlegend=False)
        return fig


def main():
    st.title("🧬 Enhanced Bioinformatics Analysis Platform")
    st.markdown("### Comprehensive Pipeline for mRNA and Genomic Analysis")
    
    # Initialize analysis runner
    if 'runner' not in st.session_state:
        st.session_state.runner = EnhancedAnalysisRunner()
    
    runner = st.session_state.runner
    
    # Step 1: File Upload Section
    st.markdown("## 📁 Step 1: Upload Required Files")
    st.markdown("**Upload your genomic files to enable sample generation and analysis:**")
    
    upload_container = st.container()
    with upload_container:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="upload-section">', unsafe_allow_html=True)
            
            # FASTA file upload (multiple files)
            fasta_files = st.file_uploader(
                "📄 Upload FASTA Files (Reference Genome)",
                type=['fasta', 'fa', 'fna'],
                accept_multiple_files=True,
                help="Upload one or more FASTA files containing reference genome sequences"
            )
            
            # JSON file upload
            json_file = st.file_uploader(
                "📋 Upload JSON File (Dataset Catalog)",
                type=['json'],
                help="Upload dataset catalog JSON file"
            )
            
            # JSONL file upload
            jsonl_file = st.file_uploader(
                "📊 Upload JSONL File (Assembly Report)",
                type=['jsonl'],
                help="Upload assembly report JSONL file"
            )
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 📋 File Upload Status")
            
            # File status indicators
            if fasta_files:
                st.markdown(f'<div class="file-status file-uploaded">✅ FASTA Files: {len(fasta_files)} uploaded</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="file-status file-missing">❌ FASTA Files: Not uploaded</div>', unsafe_allow_html=True)
            
            if json_file:
                st.markdown('<div class="file-status file-uploaded">✅ JSON File: Uploaded</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="file-status file-missing">❌ JSON File: Not uploaded</div>', unsafe_allow_html=True)
            
            if jsonl_file:
                st.markdown('<div class="file-status file-uploaded">✅ JSONL File: Uploaded</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="file-status file-missing">❌ JSONL File: Not uploaded</div>', unsafe_allow_html=True)
    
    # Check if files are uploaded
    files_uploaded = fasta_files and json_file and jsonl_file
    
    if files_uploaded:
        # Save uploaded files
        saved_files = runner.save_uploaded_files(fasta_files, json_file, jsonl_file)
        
        st.success("✅ **All files uploaded successfully!**")
        st.markdown("**Files saved:**")
        for file_type, file_path in saved_files.items():
            if isinstance(file_path, list):
                st.write(f"• {file_type}: {len(file_path)} files")
            else:
                st.write(f"• {file_type}: {os.path.basename(file_path)}")
        
        # Step 2: Analysis Selection
        st.markdown("---")
        st.markdown("## 🔬 Step 2: Select Analysis Pipelines")
        
        # Analysis options with sample creation options
        analysis_options = {
            "🧪 mRNA Simulation Platform": {
                "function": runner.run_mrna_simulation,
                "description": "AI-Physics mRNA simulation using DeepChem and OpenMM",
                "has_sample_option": True,
                "sample_description": "Generate synthetic mRNA sequences for simulation"
            },
            "📊 FastQC Quality Control": {
                "function": runner.run_fastqc_analysis,
                "description": "Comprehensive sequence quality assessment",
                "has_sample_option": True,
                "sample_description": "Create sample FASTQ files for quality analysis"
            },
            "🔧 Gene Sequence Optimization": {
                "function": runner.run_gene_optimization,
                "description": "Codon optimization using DNA Chisel",
                "has_sample_option": True,
                "sample_description": "Generate sample sequences or extract from uploaded FASTA"
            },
            "💉 mRNA Vaccine Design": {
                "function": runner.run_vaccine_design,
                "description": "Advanced epitope prediction and vaccine design",
                "has_sample_option": False,
                "sample_description": "Uses uploaded genomic files"
            },
            "🔀 mRNA Folding Optimization": {
                "function": runner.run_mrna_folding,
                "description": "Structure optimization and folding prediction",
                "has_sample_option": True,
                "sample_description": "Uses example protein sequence for optimization"
            },
            "🎯 MHCflurry Integration": {
                "function": runner.run_mhcflurry_analysis,
                "description": "IEDB-focused epitope prediction",
                "has_sample_option": False,
                "sample_description": "Uses uploaded genomic files"
            },
            "🧬 Nanopore Sequencing Analysis": {
                "function": runner.run_nanopore_analysis,
                "description": "FAST5 file processing and basecalling",
                "has_sample_option": True,
                "sample_description": "Create sample FAST5 files for analysis"
            },
            "🔍 Illumina Sequencing Analysis": {
                "function": runner.run_illumina_analysis,
                "description": "Variant calling and SNP analysis",
                "has_sample_option": True,
                "sample_description": "Create sample VCF files for variant analysis"
            },
            "⚗️ PacBio SMRT Analysis": {
                "function": runner.run_pacbio_analysis,
                "description": "HiFi reads and variant detection using RDKit",
                "has_sample_option": True,
                "sample_description": "Generate sample HiFi reads for analysis"
            },
            "🗺️ Spatial Transcriptomics": {
                "function": runner.run_spatial_analysis,
                "description": "Spatial gene expression analysis",
                "has_sample_option": True,
                "sample_description": "Create sample spatial transcriptomics data"
            }
        }
        
        # Create analysis selection interface
        selected_analyses = {}
        
        st.markdown("**Select analyses and configure sample generation:**")
        
        for analysis_name, analysis_info in analysis_options.items():
            with st.expander(f"{analysis_name}", expanded=False):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**Description:** {analysis_info['description']}")
                    
                    # Checkbox to select analysis
                    select_analysis = st.checkbox(f"Run {analysis_name}", key=f"select_{analysis_name}")
                    
                    if select_analysis:
                        # Sample generation option
                        if analysis_info['has_sample_option']:
                            create_sample = st.checkbox(
                                f"Create sample data",
                                value=True,
                                help=analysis_info['sample_description'],
                                key=f"sample_{analysis_name}"
                            )
                            selected_analyses[analysis_name] = {
                                'function': analysis_info['function'],
                                'create_sample': create_sample
                            }
                        else:
                            selected_analyses[analysis_name] = {
                                'function': analysis_info['function'],
                                'create_sample': False
                            }
                
                with col2:
                    if analysis_info['has_sample_option']:
                        st.success("✅ Sample data available")
                    else:
                        st.info("📁 Uses uploaded files")
        
        # Step 3: Run Analyses
        if selected_analyses:
            st.markdown("---")
            st.markdown("## 🚀 Step 3: Run Selected Analyses")
            
            st.success(f"✅ **Selected {len(selected_analyses)} analysis(es)**")
            for analysis_name in selected_analyses.keys():
                sample_status = "with sample data" if selected_analyses[analysis_name]['create_sample'] else "with uploaded files"
                st.write(f"• {analysis_name} ({sample_status})")
            
            if st.button("🔬 **RUN ALL SELECTED ANALYSES**", type="primary", use_container_width=True):
                
                # Initialize progress tracking
                progress_bar = st.progress(0)
                status_container = st.empty()
                
                # Results containers
                results_container = st.container()
                
                with results_container:
                    st.markdown("## 📊 Analysis Results")
                    
                    for i, (analysis_name, analysis_config) in enumerate(selected_analyses.items()):
                        # Update progress
                        progress = (i + 1) / len(selected_analyses)
                        progress_bar.progress(progress)
                        status_container.info(f"Running {i+1}/{len(selected_analyses)}: {analysis_name}")
                        
                        # Run analysis
                        analysis_function = analysis_config['function']
                        create_sample = analysis_config['create_sample']
                        
                        try:
                            # Call function with or without create_sample parameter
                            if analysis_name in ["💉 mRNA Vaccine Design", "🎯 MHCflurry Integration"]:
                                # These don't use create_sample
                                stats, plot = analysis_function()
                            else:
                                # These use create_sample parameter
                                stats, plot = analysis_function(create_sample=create_sample)
                            
                            if stats and plot:
                                # Create expandable section for each analysis
                                with st.expander(f"✅ {analysis_name} - Results", expanded=True):
                                    
                                    # Display statistics in columns
                                    st.markdown(f"### 📈 {analysis_name} Statistics")
                                    
                                    # Create metric columns
                                    cols = st.columns(3)
                                    stat_items = list(stats.items())
                                    
                                    for idx, (key, value) in enumerate(stat_items):
                                        col_idx = idx % 3
                                        with cols[col_idx]:
                                            st.metric(key, value)
                                    
                                    # Display plot
                                    st.markdown(f"### 📊 {analysis_name} Visualizations")
                                    st.plotly_chart(plot, use_container_width=True)
                                    
                                    # Success message
                                    st.success(f"✅ {analysis_name} completed successfully!")
                            
                            else:
                                st.error(f"❌ {analysis_name} failed to generate results")
                        
                        except Exception as e:
                            st.error(f"❌ {analysis_name} failed: {str(e)}")
                    
                    # Final completion message
                    progress_bar.progress(1.0)
                    status_container.success("🎉 All analyses completed!")
                    
                    # Summary statistics
                    if runner.results:
                        st.markdown("---")
                        st.markdown("## 🎯 Overall Summary")
                        
                        summary_cols = st.columns(4)
                        
                        with summary_cols[0]:
                            st.metric("Total Analyses", len(runner.results))
                        
                        with summary_cols[1]:
                            successful = sum(1 for r in runner.results.values() if r['status'] == 'success')
                            st.metric("Successful", successful)
                        
                        with summary_cols[2]:
                            failed = len(runner.results) - successful
                            st.metric("Failed", failed)
                        
                        with summary_cols[3]:
                            success_rate = (successful / len(runner.results)) * 100 if runner.results else 0
                            st.metric("Success Rate", f"{success_rate:.1f}%")
                        
                        # Download section
                        st.markdown("### 📥 Download Results")
                        
                        # Create downloadable summary
                        summary_data = {
                            "timestamp": datetime.now().isoformat(),
                            "total_analyses": len(runner.results),
                            "successful_analyses": successful,
                            "uploaded_files": {
                                "fasta_files": len(fasta_files),
                                "json_file": json_file.name if json_file else None,
                                "jsonl_file": jsonl_file.name if jsonl_file else None
                            },
                            "analysis_results": {name: result['stats'] for name, result in runner.results.items()}
                        }
                        
                        summary_json = json.dumps(summary_data, indent=2, default=str)
                        
                        st.download_button(
                            label="📊 Download Summary Report (JSON)",
                            data=summary_json,
                            file_name=f"bioinformatics_analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
        
        else:
            st.info("👆 **Please select at least one analysis from the options above**")
    
    else:
        # Show instructions when files are not uploaded
        st.markdown("---")
        st.markdown("## 📋 Instructions")
        
        st.info("**Please upload the required files to proceed with analysis:**")
        
        instruction_cols = st.columns(3)
        
        with instruction_cols[0]:
            st.markdown("""
            ### 📄 **FASTA Files**
            - Reference genome sequences
            - Multiple files supported
            - Formats: .fasta, .fa, .fna
            - Used for sequence analysis
            """)
        
        with instruction_cols[1]:
            st.markdown("""
            ### 📋 **JSON File**
            - Dataset catalog information
            - Contains organism metadata
            - Format: .json
            - Required for all analyses
            """)
        
        with instruction_cols[2]:
            st.markdown("""
            ### 📊 **JSONL File**
            - Assembly report data
            - Genome assembly information
            - Format: .jsonl
            - Required for pipeline setup
            """)
        
        # Show available analyses preview
        st.markdown("---")
        st.markdown("## 🔬 Available Analysis Pipelines")
        st.markdown("**Once files are uploaded, you can run these analyses:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🧬 **mRNA & Gene Analysis**
            - **🧪 mRNA Simulation Platform**: AI-Physics simulation using DeepChem
            - **🔧 Gene Sequence Optimization**: Codon optimization with DNA Chisel
            - **💉 mRNA Vaccine Design**: Epitope prediction and vaccine design
            - **🔀 mRNA Folding Optimization**: Structure optimization and folding
            - **🎯 MHCflurry Integration**: IEDB-focused epitope prediction
            """)
        
        with col2:
            st.markdown("""
            ### 🔍 **Sequencing Technologies**
            - **📊 FastQC Quality Control**: Comprehensive sequence QC
            - **🧬 Nanopore Sequencing**: FAST5 processing and basecalling
            - **🔍 Illumina Sequencing**: Variant calling and SNP analysis
            - **⚗️ PacBio SMRT Analysis**: HiFi reads and variant detection
            - **🗺️ Spatial Transcriptomics**: Spatial gene expression analysis
            """)
        
        # Features highlights
        st.markdown("---")
        st.markdown("## ✨ Platform Features")
        
        feature_cols = st.columns(3)
        
        with feature_cols[0]:
            st.markdown("""
            ### 🎯 **Automatic Sample Generation**
            - No need for additional sample files
            - Realistic synthetic data creation
            - Option to use uploaded files or samples
            - Ready-to-run analyses
            """)
        
        with feature_cols[1]:
            st.markdown("""
            ### 📊 **Comprehensive Statistics**
            - Detailed metrics for each analysis
            - Interactive Plotly visualizations
            - Professional result displays
            - Downloadable JSON reports
            """)
        
        with feature_cols[2]:
            st.markdown("""
            ### 🔬 **Multiple Pipeline Support**
            - 10 different analysis types
            - Integrated bioinformatics workflows
            - Sample data or file-based analysis
            - Professional research results
            """)
        
        # Sample file information
        st.markdown("---")
        st.markdown("## 📁 Sample File Requirements")
        
        with st.expander("📖 **Click to see example file formats**", expanded=False):
            st.markdown("""
            ### FASTA File Example:
            ```
            >chromosome_1
            ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG
            CGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATC
            >chromosome_2
            GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA
            ```
            
            ### JSON File Example:
            ```json
            {
                "organism": "Homo sapiens",
                "assembly": "GRCh38",
                "version": "p14",
                "description": "Human reference genome"
            }
            ```
            
            ### JSONL File Example:
            ```json
            {"assemblyInfo": {"assemblyName": "GRCh38.p14", "description": "Human reference genome"}}
            ```
            """)


if __name__ == "__main__":
    main()