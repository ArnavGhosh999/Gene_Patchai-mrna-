#!/usr/bin/env python3
"""
IEDB-Focused mRNA Vaccine Epitope Prediction Pipeline
====================================================

This version focuses on IEDB API integration with robust fallback methods
when MHCflurry models are not available on Windows.

Requirements:
- pip install biopython pandas numpy requests
"""

import os
import json
import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
import requests
import time
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import urllib.parse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IEDBmRNAPredictor:
    """
    IEDB-focused pipeline for mRNA vaccine epitope prediction
    """
    
    def __init__(self, genome_path: str, catalog_path: str, assembly_report_path: str):
        """
        Initialize the epitope predictor with genomic data files
        """
        self.genome_path = genome_path
        self.catalog_path = catalog_path
        self.assembly_report_path = assembly_report_path
        
        # Common HLA alleles for population coverage
        self.common_hla_alleles = [
            'HLA-A*02:01', 'HLA-A*01:01', 'HLA-A*03:01', 'HLA-A*24:02',
            'HLA-B*07:02', 'HLA-B*08:01', 'HLA-B*35:01', 'HLA-B*44:02',
            'HLA-C*07:02', 'HLA-C*07:01', 'HLA-C*06:02', 'HLA-C*03:04'
        ]
        
        # Try to load MHCflurry if available, otherwise continue without it
        self.mhc_predictor = None
        self.load_mhcflurry()
        
        self.load_genomic_data()
    
    def load_mhcflurry(self):
        """Try to load MHCflurry, continue gracefully if not available"""
        try:
            from mhcflurry import Class1AffinityPredictor
            self.mhc_predictor = Class1AffinityPredictor.load()
            logger.info("MHCflurry models loaded successfully")
        except Exception as e:
            logger.warning(f"MHCflurry not available: {e}")
            logger.info("Will use IEDB API and scoring methods instead")
    
    def load_genomic_data(self):
        """Load and parse genomic data files"""
        logger.info("Loading genomic data files...")
        
        try:
            # Load dataset catalog
            with open(self.catalog_path, 'r') as f:
                self.catalog = json.load(f)
            
            # Load assembly report
            self.assembly_data = []
            with open(self.assembly_report_path, 'r') as f:
                for line in f:
                    if line.strip():
                        self.assembly_data.append(json.loads(line.strip()))
            
            logger.info(f"Loaded catalog with {len(self.catalog)} entries")
            logger.info(f"Loaded assembly data with {len(self.assembly_data)} records")
        except Exception as e:
            logger.error(f"Error loading genomic data: {e}")
            raise
    
    def translate_sequence(self, dna_sequence: str, reading_frame: int = 0) -> str:
        """Translate DNA sequence to protein"""
        if reading_frame > 2:
            raise ValueError("Reading frame must be 0, 1, or 2")
        
        adjusted_seq = dna_sequence[reading_frame:]
        remainder = len(adjusted_seq) % 3
        if remainder:
            adjusted_seq = adjusted_seq[:-remainder]
        
        if len(adjusted_seq) < 3:
            return ""
        
        protein_seq = Seq(adjusted_seq).translate()
        return str(protein_seq)
    
    def extract_mrna_sequences(self, target_genes: List[str] = None, max_sequences: int = 3) -> Dict[str, str]:
        """Extract mRNA sequences from genomic data"""
        mrna_sequences = {}
        
        logger.info("Extracting mRNA sequences from genome...")
        
        try:
            sequence_count = 0
            for record in SeqIO.parse(self.genome_path, "fasta"):
                if sequence_count >= max_sequences:
                    break
                
                seq_id = record.id
                sequence = str(record.seq)
                
                # Skip very short sequences
                if len(sequence) < 100:
                    continue
                
                if target_genes is None or seq_id in target_genes:
                    # Take a reasonable chunk for processing (not entire chromosome)
                    chunk_size = min(10000, len(sequence))
                    mrna_sequences[seq_id] = sequence[:chunk_size]
                    sequence_count += 1
            
            logger.info(f"Extracted {len(mrna_sequences)} mRNA sequences")
            return mrna_sequences
        except Exception as e:
            logger.error(f"Error extracting mRNA sequences: {e}")
            return {}
    
    def generate_peptides(self, protein_sequence: str, peptide_lengths: List[int] = [8, 9, 10, 11]) -> List[str]:
        """Generate overlapping peptides from protein sequence"""
        peptides = []
        
        for length in peptide_lengths:
            for i in range(len(protein_sequence) - length + 1):
                peptide = protein_sequence[i:i + length]
                # Filter out peptides with stop codons or unusual amino acids
                if '*' not in peptide and 'X' not in peptide and len(peptide) == length:
                    peptides.append(peptide)
        
        return list(set(peptides))  # Remove duplicates
    
    def query_iedb_mhci(self, peptides: List[str], method: str = 'netmhcpan_ba', 
                        alleles: List[str] = None) -> Dict:
        """
        Query IEDB API for MHC Class I binding predictions
        """
        if alleles is None:
            # Use a subset of alleles that IEDB supports
            alleles = ['HLA-A*02:01', 'HLA-A*01:01', 'HLA-B*07:02']
        
        # Limit peptides for API call
        limited_peptides = peptides[:20]  # IEDB has limits
        
        url = "http://tools-cluster-interface.iedb.org/tools_api/mhci/"
        
        # Format alleles for IEDB
        allele_string = ','.join(alleles)
        
        data = {
            'method': method,
            'sequence_text': '\n'.join(limited_peptides),
            'allele': allele_string,
            'length': '9'  # Focus on 9-mers for now
        }
        
        try:
            logger.info(f"Querying IEDB API for {len(limited_peptides)} peptides...")
            response = requests.post(url, data=data, timeout=60)
            
            if response.status_code == 200:
                return {
                    'status': 'success', 
                    'data': response.text,
                    'peptides_queried': limited_peptides,
                    'alleles_queried': alleles
                }
            else:
                return {
                    'status': 'error', 
                    'message': f"HTTP {response.status_code}",
                    'response': response.text[:500]
                }
        except requests.RequestException as e:
            logger.error(f"IEDB API request failed: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def parse_iedb_results(self, iedb_response: Dict) -> pd.DataFrame:
        """Parse IEDB API response into DataFrame"""
        if iedb_response['status'] != 'success':
            logger.warning("IEDB query failed, returning empty DataFrame")
            return pd.DataFrame()
        
        try:
            lines = iedb_response['data'].strip().split('\n')
            
            # Find header line
            header_idx = 0
            for i, line in enumerate(lines):
                if line.startswith('allele') or 'peptide' in line.lower():
                    header_idx = i
                    break
            
            if header_idx < len(lines):
                # Parse tabular data
                header = lines[header_idx].split('\t')
                data_rows = []
                
                for line in lines[header_idx + 1:]:
                    if line.strip() and not line.startswith('#'):
                        values = line.split('\t')
                        if len(values) >= len(header):
                            data_rows.append(values[:len(header)])
                
                if data_rows:
                    df = pd.DataFrame(data_rows, columns=header)
                    logger.info(f"Parsed {len(df)} IEDB predictions")
                    return df
            
            logger.warning("Could not parse IEDB response format")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error parsing IEDB results: {e}")
            return pd.DataFrame()
    
    def score_peptides_simple(self, peptides: List[str]) -> pd.DataFrame:
        """
        Simple peptide scoring when other methods aren't available
        """
        logger.info("Using simple peptide scoring method")
        
        # Amino acid properties for scoring
        hydrophobic = {'A', 'I', 'L', 'M', 'F', 'P', 'W', 'Y', 'V'}
        charged = {'D', 'E', 'K', 'R'}
        polar = {'N', 'Q', 'S', 'T', 'C', 'G', 'H'}
        
        results = []
        
        for peptide in peptides:
            # Calculate composition scores
            hydrophobic_score = sum(1 for aa in peptide if aa in hydrophobic) / len(peptide)
            charged_score = sum(1 for aa in peptide if aa in charged) / len(peptide)
            polar_score = sum(1 for aa in peptide if aa in polar) / len(peptide)
            
            # Simple immunogenicity score (not scientifically validated!)
            immunogenicity_score = (hydrophobic_score * 0.4 + 
                                   charged_score * 0.3 + 
                                   polar_score * 0.3)
            
            # Convert to mock percentile (lower is better for binding)
            mock_percentile = 100 * (1 - immunogenicity_score)
            
            results.append({
                'peptide': peptide,
                'length': len(peptide),
                'hydrophobic_fraction': hydrophobic_score,
                'charged_fraction': charged_score,
                'polar_fraction': polar_score,
                'immunogenicity_score': immunogenicity_score,
                'mock_binding_percentile': mock_percentile,
                'prediction_method': 'simple_scoring'
            })
        
        df = pd.DataFrame(results)
        
        # Add binding categories
        df['binding_category'] = pd.cut(
            df['mock_binding_percentile'],
            bins=[0, 10, 30, 70, 100],
            labels=['Strong', 'Moderate', 'Weak', 'Non-binder']
        )
        
        return df
    
    def run_epitope_pipeline(self, target_genes: List[str] = None, 
                           output_dir: str = 'iedb_epitope_results') -> Dict:
        """
        Run the complete epitope prediction pipeline
        """
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        logger.info("Starting IEDB-focused epitope prediction pipeline...")
        
        try:
            # Step 1: Extract mRNA sequences
            mrna_sequences = self.extract_mrna_sequences(target_genes, max_sequences=3)
            
            if not mrna_sequences:
                logger.error("No mRNA sequences extracted")
                return {'error': 'No sequences found'}
            
            # Step 2: Translate and generate peptides
            all_peptides = []
            sequence_info = []
            
            for seq_id, mrna_seq in mrna_sequences.items():
                logger.info(f"Processing sequence: {seq_id}")
                
                # Try multiple reading frames
                for frame in range(3):
                    try:
                        protein_seq = self.translate_sequence(mrna_seq, frame)
                        if len(protein_seq) > 10:  # Skip very short proteins
                            peptides = self.generate_peptides(protein_seq)
                            
                            for peptide in peptides[:50]:  # Limit per sequence
                                all_peptides.append(peptide)
                                sequence_info.append({
                                    'sequence_id': seq_id,
                                    'reading_frame': frame,
                                    'peptide': peptide,
                                    'protein_length': len(protein_seq)
                                })
                    except Exception as e:
                        logger.warning(f"Error translating {seq_id} frame {frame}: {e}")
            
            # Remove duplicates
            unique_peptides = list(dict.fromkeys(all_peptides))
            
            # Limit total peptides for processing
            if len(unique_peptides) > 200:
                unique_peptides = unique_peptides[:200]
            
            logger.info(f"Generated {len(unique_peptides)} unique peptides for analysis")
            
            # Step 3: IEDB API prediction
            iedb_response = self.query_iedb_mhci(unique_peptides)
            iedb_df = self.parse_iedb_results(iedb_response)
            
            # Step 4: Simple scoring as backup
            simple_scores = self.score_peptides_simple(unique_peptides)
            
            # Step 5: MHCflurry prediction if available
            mhcflurry_df = pd.DataFrame()
            if self.mhc_predictor is not None:
                try:
                    # Try MHCflurry prediction
                    predictions = self.mhc_predictor.predict(
                        peptides=unique_peptides[:50],  # Limit for stability
                        alleles=['HLA-A*02:01', 'HLA-A*01:01']
                    )
                    mhcflurry_df = pd.DataFrame(predictions)
                    logger.info("MHCflurry predictions completed")
                except Exception as e:
                    logger.warning(f"MHCflurry prediction failed: {e}")
            
            # Compile results
            results = {
                'summary': {
                    'total_sequences': len(mrna_sequences),
                    'total_peptides': len(unique_peptides),
                    'iedb_predictions': len(iedb_df),
                    'simple_scores': len(simple_scores),
                    'mhcflurry_predictions': len(mhcflurry_df)
                },
                'sequence_info': sequence_info[:50],  # Limit for JSON
                'iedb_results': iedb_response,
                'peptide_analysis': {
                    'top_peptides_simple': simple_scores.nlargest(10, 'immunogenicity_score').to_dict('records'),
                    'length_distribution': simple_scores['length'].value_counts().to_dict()
                }
            }
            
            # Save results
            with open(f"{output_dir}/epitope_analysis.json", 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            simple_scores.to_csv(f"{output_dir}/peptide_scores.csv", index=False)
            
            if not iedb_df.empty:
                iedb_df.to_csv(f"{output_dir}/iedb_predictions.csv", index=False)
            
            if not mhcflurry_df.empty:
                mhcflurry_df.to_csv(f"{output_dir}/mhcflurry_predictions.csv", index=False)
            
            logger.info(f"Pipeline completed successfully! Results saved to {output_dir}/")
            
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return {'error': str(e)}
    
    def generate_report(self, results: Dict, output_file: str = None) -> str:
        """Generate a summary report"""
        if 'error' in results:
            report = f"Pipeline Error: {results['error']}"
        else:
            summary = results['summary']
            report = f"""
mRNA Vaccine Epitope Analysis Report
===================================

PIPELINE SUMMARY
---------------
• Processed {summary['total_sequences']} mRNA sequences
• Generated {summary['total_peptides']} unique peptides
• IEDB API predictions: {summary['iedb_predictions']} results
• Simple scoring: {summary['simple_scores']} peptides scored
• MHCflurry predictions: {summary['mhcflurry_predictions']} results

TOP PEPTIDE CANDIDATES (Simple Scoring)
--------------------------------------
"""
            
            if 'peptide_analysis' in results:
                for i, peptide in enumerate(results['peptide_analysis']['top_peptides_simple'][:5], 1):
                    report += f"{i}. {peptide['peptide']} (Score: {peptide['immunogenicity_score']:.3f})\n"
            
            report += f"""

PEPTIDE LENGTH DISTRIBUTION
--------------------------
"""
            if 'peptide_analysis' in results:
                for length, count in results['peptide_analysis']['length_distribution'].items():
                    report += f"• Length {length}: {count} peptides\n"
            
            report += f"""

RECOMMENDATIONS
--------------
• Validate top candidates experimentally
• Consider IEDB results for final selection
• Optimize mRNA codons for expression
• Test immunogenicity in appropriate models

NEXT STEPS
----------
1. Download MHCflurry models for more accurate predictions
2. Expand HLA allele coverage for population analysis
3. Include B-cell epitope prediction
4. Design multi-epitope mRNA constructs
"""
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
        
        return report


def main():
    """Main function to run the pipeline"""
    # File paths
    genome_path = "ncbi_dataset/ncbi_dataset/data/GCA_000001405.29_GRCh38.p14_genomic.fna"
    catalog_path = "ncbi_dataset/ncbi_dataset/data/dataset_catalog.json"
    assembly_report_path = "ncbi_dataset/ncbi_dataset/data/assembly_data_report.jsonl"
    
    # Check if files exist
    for path in [genome_path, catalog_path, assembly_report_path]:
        if not os.path.exists(path):
            logger.error(f"File not found: {path}")
            return
    
    try:
        # Initialize predictor
        predictor = IEDBmRNAPredictor(genome_path, catalog_path, assembly_report_path)
        
        # Run pipeline
        results = predictor.run_epitope_pipeline()
        
        # Generate report
        report = predictor.generate_report(results, "iedb_epitope_report.txt")
        print(report)
        
        logger.info("IEDB-focused pipeline completed!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()