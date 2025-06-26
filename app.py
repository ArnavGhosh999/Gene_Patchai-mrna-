#!/usr/bin/env python3
"""
Bioinformatics Analysis Platform - Streamlit App
================================================

Unified interface for multiple bioinformatics pipelines:
- mRNA Simulation (biosimai.py)
- FastQC Analysis (fastqc.py) 
- Gene Optimization (gene_optimizer.py)
- mRNA Vaccine Design (iedb.py)
- mRNA Folding Optimization (linear_design.py)
- MHCflurry Integration (mhcflurry_download.py)
- Nanopore Analysis (nanopore.py)
- Illumina Sequencing (sequencing.py)
- PacBio SMRT (smrt.py)
- Spatial Transcriptomics (visium.py)

Requirements:
pip install streamlit pandas numpy matplotlib seaborn biopython
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

# Import the pipeline modules (assuming they're in the same directory)
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
    page_title="Bioinformatics Analysis Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🧬 Bioinformatics Analysis Platform")
    st.markdown("### Comprehensive Pipeline for mRNA and Genomic Analysis")
    
    # Sidebar for navigation
    st.sidebar.title("Analysis Options")
    
    # File Upload Section
    st.sidebar.header("📁 File Upload")
    
    # Upload required files (no size limit)
    fasta_file = st.sidebar.file_uploader(
        "Upload FASTA file (Reference Genome)",
        type=['fasta', 'fa', 'fna'],
        help="Reference genome sequence file (no size limit)"
    )
    
    json_file = st.sidebar.file_uploader(
        "Upload JSON file (Dataset Catalog)",
        type=['json'],
        help="Dataset catalog JSON file (no size limit)"
    )
    
    jsonl_file = st.sidebar.file_uploader(
        "Upload JSONL file (Assembly Report)",
        type=['jsonl'],
        help="Assembly report JSONL file (no size limit)"
    )
    
    # Additional file uploads for specific analyses
    additional_files = {}
    
    # Analysis Selection
    st.sidebar.header("🔬 Analysis Pipeline")
    
    analysis_options = {
        "mRNA Simulation Platform": {
            "module": "biosimai",
            "description": "AI-Physics mRNA simulation using DeepChem and OpenMM",
            "icon": "🧪",
            "additional_files": ["FASTQ files (optional)"]
        },
        "FastQC Quality Control": {
            "module": "fastqc",
            "description": "Comprehensive sequence quality assessment",
            "icon": "📊",
            "additional_files": ["FASTQ files for QC"]
        },
        "Gene Sequence Optimization": {
            "module": "gene_optimizer",
            "description": "Codon optimization using DNA Chisel",
            "icon": "🔧",
            "additional_files": ["Input sequences (FASTA)"]
        },
        "mRNA Vaccine Design": {
            "module": "iedb",
            "description": "Advanced epitope prediction and vaccine design",
            "icon": "💉",
            "additional_files": []
        },
        "mRNA Folding Optimization": {
            "module": "linear_design",
            "description": "Structure optimization and folding prediction",
            "icon": "🔀",
            "additional_files": []
        },
        "MHCflurry Integration": {
            "module": "mhcflurry_download",
            "description": "IEDB-focused epitope prediction",
            "icon": "🎯",
            "additional_files": []
        },
        "Nanopore Sequencing Analysis": {
            "module": "nanopore",
            "description": "FAST5 file processing and basecalling",
            "icon": "🧬",
            "additional_files": ["FAST5 files"]
        },
        "Illumina Sequencing Analysis": {
            "module": "sequencing",
            "description": "Variant calling and SNP analysis",
            "icon": "🔍",
            "additional_files": ["VCF files", "InterOp metrics"]
        },
        "PacBio SMRT Analysis": {
            "module": "smrt",
            "description": "HiFi reads and variant detection using RDKit",
            "icon": "🔬",
            "additional_files": ["HiFi reads (JSON)"]
        },
        "Spatial Transcriptomics": {
            "module": "visium",
            "description": "Spatial gene expression analysis",
            "icon": "🗺️",
            "additional_files": ["Expression matrix", "Spatial coordinates"]
        }
    }
    
    selected_analysis = st.sidebar.selectbox(
        "Choose Analysis Pipeline:",
        list(analysis_options.keys()),
        help="Select the type of analysis you want to perform"
    )
    
    # Display selected analysis info
    analysis_info = analysis_options[selected_analysis]
    st.sidebar.info(f"**{analysis_info['icon']} {selected_analysis}**\n\n{analysis_info['description']}")
    
    # Main content area
    st.header(f"🧬 Bioinformatics Analysis Platform")
    st.markdown("### Comprehensive Pipeline for mRNA and Genomic Analysis")
    
    # Show file upload status
    files_uploaded = fasta_file is not None and json_file is not None and jsonl_file is not None
    
    if not files_uploaded:
        st.warning("⚠️ Please upload all required files (FASTA, JSON, JSONL) in the sidebar to proceed.")
        st.info("""
        **Required Files:**
        - **FASTA file**: Reference genome sequence
        - **JSON file**: Dataset catalog
        - **JSONL file**: Assembly report
        
        **OR** choose an analysis that can generate sample data (most analyses support this option).
        """)
        
        # Show available analyses even without files
        st.subheader("🔬 Available Analysis Pipelines")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🧬 mRNA & Gene Analysis:**
            - mRNA Simulation Platform
            - Gene Sequence Optimization
            - mRNA Vaccine Design
            - mRNA Folding Optimization
            
            **📊 Quality Control:**
            - FastQC Quality Control
            - MHCflurry Integration
            """)
        
        with col2:
            st.markdown("""
            **🔍 Sequencing Technologies:**
            - Nanopore Sequencing Analysis
            - Illumina Sequencing Analysis
            - PacBio SMRT Analysis
            
            **🎯 Specialized:**
            - Spatial Transcriptomics
            """)
        
        st.info("💡 **Tip:** Most analyses can generate sample data for testing - just upload the files and select 'Create sample' in the parameters!")
    
    else:
        st.success("✅ All required files uploaded! Ready for analysis.")
        
        # Save uploaded files to temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Save files
        fasta_path = save_uploaded_file(fasta_file, temp_dir, "reference.fasta")
        json_path = save_uploaded_file(json_file, temp_dir, "dataset_catalog.json")
        jsonl_path = save_uploaded_file(jsonl_file, temp_dir, "assembly_report.jsonl")
        
        st.info(f"📁 Files saved to temporary directory: `{temp_dir}`")
    
    # Analysis Selection - Checkbox style for multiple selections
    st.header("🔬 Choose Your Analysis Pipeline(s)")
    st.markdown("**Select one or more analyses to run:**")
    
    # Create columns for analysis selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Recommended analyses section
        st.markdown("""
        <div style='background-color: #2d5aa0; padding: 15px; border-radius: 8px; margin-bottom: 20px;'>
            <h4 style='margin: 0; color: white;'>🔍 Recommended: mRNA-focused analyses</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Create checkboxes for each analysis
        selected_analyses = []
        
        # Group 1: mRNA & Gene Analysis (Recommended)
        st.markdown("### 🧬 mRNA & Gene Analysis")
        
        if st.checkbox("🧪 mRNA Simulation Platform", help="AI-Physics mRNA simulation using DeepChem and OpenMM"):
            selected_analyses.append("mRNA Simulation Platform")
            
        if st.checkbox("🔧 Gene Sequence Optimization", help="Codon optimization using DNA Chisel"):
            selected_analyses.append("Gene Sequence Optimization")
            
        if st.checkbox("💉 mRNA Vaccine Design", help="Advanced epitope prediction and vaccine design"):
            selected_analyses.append("mRNA Vaccine Design")
            
        if st.checkbox("🔀 mRNA Folding Optimization", help="Structure optimization and folding prediction"):
            selected_analyses.append("mRNA Folding Optimization")
        
        # Group 2: Quality Control
        st.markdown("### 📊 Quality Control")
        
        if st.checkbox("📈 FastQC Quality Control", help="Comprehensive sequence quality assessment"):
            selected_analyses.append("FastQC Quality Control")
            
        if st.checkbox("🎯 MHCflurry Integration", help="IEDB-focused epitope prediction"):
            selected_analyses.append("MHCflurry Integration")
        
        # Group 3: Sequencing Technologies
        st.markdown("### 🔍 Sequencing Technologies")
        
        if st.checkbox("🧬 Nanopore Sequencing Analysis", help="FAST5 file processing and basecalling"):
            selected_analyses.append("Nanopore Sequencing Analysis")
            
        if st.checkbox("🔬 Illumina Sequencing Analysis", help="Variant calling and SNP analysis"):
            selected_analyses.append("Illumina Sequencing Analysis")
            
        if st.checkbox("⚗️ PacBio SMRT Analysis", help="HiFi reads and variant detection using RDKit"):
            selected_analyses.append("PacBio SMRT Analysis")
        
        # Group 4: Specialized Analysis
        st.markdown("### 🎯 Specialized Analysis")
        
        if st.checkbox("🗺️ Spatial Transcriptomics", help="Spatial gene expression analysis"):
            selected_analyses.append("Spatial Transcriptomics")
        
        # Show selected analyses
        if selected_analyses:
            st.success(f"✅ **Selected {len(selected_analyses)} analysis(es):**")
            for i, analysis in enumerate(selected_analyses, 1):
                analysis_info = analysis_options[analysis]
                st.write(f"{i}. {analysis_info['icon']} {analysis}")
        else:
            st.info("💡 Please select at least one analysis to proceed.")
        
        # Big prominent RUN button for multiple analyses (no parameters)
        st.markdown("---")
        
        if selected_analyses:
            # Check if analysis can run
            files_uploaded = fasta_file is not None and json_file is not None and jsonl_file is not None
            
            if files_uploaded:
                if st.button(f"🚀 **RUN SELECTED ANALYSES ({len(selected_analyses)})**", 
                            type="primary", 
                            help=f"Start {len(selected_analyses)} selected analysis(es)",
                            use_container_width=True):
                    
                    # Prepare file paths
                    temp_dir = tempfile.mkdtemp()
                    fasta_path = save_uploaded_file(fasta_file, temp_dir, "reference.fasta")
                    json_path = save_uploaded_file(json_file, temp_dir, "dataset_catalog.json")
                    jsonl_path = save_uploaded_file(jsonl_file, temp_dir, "assembly_report.jsonl")
                    
                    # Run multiple analyses with default parameters
                    run_multiple_analyses_simple(selected_analyses, fasta_path, json_path, jsonl_path, temp_dir)
            else:
                # Option to run with sample data
                col_a, col_b = st.columns(2)
                
                with col_a:
                    if st.button(f"🧪 **RUN WITH SAMPLE DATA ({len(selected_analyses)})**", 
                                type="secondary", 
                                help="Generate sample data and run analyses",
                                use_container_width=True):
                        temp_dir = tempfile.mkdtemp()
                        run_multiple_analyses_simple(selected_analyses, None, None, None, temp_dir)
                
                with col_b:
                    st.button(f"🚀 **RUN WITH YOUR FILES**", 
                             disabled=True, 
                             help="Upload FASTA, JSON, and JSONL files first",
                             use_container_width=True)
        else:
            st.button("🚀 **SELECT ANALYSES TO RUN**", 
                     disabled=True, 
                     help="Select at least one analysis first",
                     use_container_width=True)
    
    with col2:
        st.header("📋 Analysis Status")
        
        # Create status container
        status_container = st.container()
        
        with status_container:
            if 'analysis_status' not in st.session_state:
                st.session_state.analysis_status = "Ready"
            
            status_color = {
                "Ready": "🟢",
                "Running": "🟡", 
                "Complete": "✅",
                "Error": "❌"
            }
            
            st.markdown(f"**Status:** {status_color.get(st.session_state.analysis_status, '⚪')} {st.session_state.analysis_status}")
            
            # Progress information
            if 'analysis_progress' in st.session_state:
                st.progress(st.session_state.analysis_progress)
            
            # Results summary
            if 'analysis_results' in st.session_state:
                st.subheader("📊 Results Summary")
                results = st.session_state.analysis_results
                for key, value in results.items():
                    st.metric(key, value)
        
        # Help section
        st.header("❓ Help & Documentation")
        
        help_info = {
            "File Formats": """
            - **FASTA**: Standard sequence format (.fasta, .fa, .fna)
            - **JSON**: JavaScript Object Notation (.json)
            - **JSONL**: JSON Lines format (.jsonl)
            """,
            "Analysis Types": """
            Each pipeline serves different purposes:
            - **Simulation**: Molecular dynamics and prediction
            - **QC**: Quality control and assessment
            - **Optimization**: Sequence improvement
            - **Design**: Therapeutic development
            """,
            "Output": """
            Results are saved in the output directory:
            - Analysis reports (TXT, JSON)
            - Visualization plots (PNG)
            - Processed data files
            - Summary statistics
            """
        }
        
        for section, content in help_info.items():
            with st.expander(f"📖 {section}"):
                st.markdown(content)

def save_uploaded_file(uploaded_file, temp_dir, filename):
    """Save uploaded file to temporary directory"""
    file_path = os.path.join(temp_dir, filename)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def handle_additional_files(analysis_name, analysis_info, temp_dir):
    """Handle additional file uploads based on analysis type"""
    additional_files = {}
    
    if analysis_info['additional_files']:
        st.subheader("📎 Additional Files")
        
        for file_type in analysis_info['additional_files']:
            if "FASTQ" in file_type:
                files = st.file_uploader(
                    f"Upload {file_type}",
                    type=['fastq', 'fq', 'fastq.gz'],
                    accept_multiple_files=True,
                    help="No file size limit"
                )
                if files:
                    additional_files[file_type] = []
                    for i, file in enumerate(files):
                        path = save_uploaded_file(file, temp_dir, f"file_{i}.fastq")
                        additional_files[file_type].append(path)
            
            elif "FAST5" in file_type:
                files = st.file_uploader(
                    f"Upload {file_type}",
                    type=['fast5'],
                    accept_multiple_files=True,
                    help="No file size limit"
                )
                if files:
                    additional_files[file_type] = []
                    for i, file in enumerate(files):
                        path = save_uploaded_file(file, temp_dir, f"file_{i}.fast5")
                        additional_files[file_type].append(path)
            
            elif "VCF" in file_type:
                file = st.file_uploader(
                    f"Upload {file_type}",
                    type=['vcf'],
                    help="No file size limit"
                )
                if file:
                    additional_files[file_type] = save_uploaded_file(file, temp_dir, "variants.vcf")
            
            else:
                file = st.file_uploader(
                    f"Upload {file_type}",
                    help="No file size limit"
                )
                if file:
                    additional_files[file_type] = save_uploaded_file(file, temp_dir, f"additional_{file_type}")
    
    return additional_files

def display_multiple_results(all_results, output_dir):
    """Display results from multiple analyses with enhanced visualization"""
    st.header("📊 Multi-Analysis Results Dashboard")
    
    # Summary statistics
    total_analyses = len(all_results)
    successful_analyses = sum(1 for result in all_results.values() if "error" not in result)
    failed_analyses = total_analyses - successful_analyses
    
    # Top-level summary with better styling
    st.markdown("### 🎯 Overall Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("📊 Total Analyses", total_analyses)
    col2.metric("✅ Successful", successful_analyses, 
                delta=f"{(successful_analyses/total_analyses)*100:.1f}%" if total_analyses > 0 else "0%")
    col3.metric("❌ Failed", failed_analyses, 
                delta=f"-{(failed_analyses/total_analyses)*100:.1f}%" if failed_analyses > 0 else "0%")
    col4.metric("📈 Success Rate", f"{(successful_analyses/total_analyses)*100:.1f}%" if total_analyses > 0 else "0%")
    
    # Create main tabs for different views
    tab1, tab2, tab3 = st.tabs(["📋 **Individual Results**", "📁 **All Files**", "⬇️ **Bulk Downloads**"])
    
    with tab1:
        st.markdown("### 📋 Individual Analysis Results")
        
        # Display results for each analysis
        for i, (analysis_name, result) in enumerate(all_results.items(), 1):
            
            # Create a nice header for each analysis
            if "error" not in result:
                status_icon = "✅"
                status_color = "green"
            else:
                status_icon = "❌"
                status_color = "red"
            
            st.markdown(f"""
            <div style='background-color: #262730; padding: 15px; border-radius: 8px; margin: 10px 0; border-left: 4px solid {status_color};'>
                <h4 style='margin: 0; color: white;'>{status_icon} {i}. {analysis_name}</h4>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander(f"View {analysis_name} Details", expanded=False):
                if "error" in result:
                    st.error(f"❌ Analysis failed: {result['error']}")
                else:
                    # Create sub-tabs for each analysis
                    sub_tab1, sub_tab2 = st.tabs(["📊 Metrics", "📄 Raw Data"])
                    
                    with sub_tab1:
                        # Display metrics in columns
                        metrics = list(result.items())
                        if len(metrics) >= 3:
                            sub_cols = st.columns(3)
                            for j, (key, value) in enumerate(metrics):
                                sub_cols[j % 3].metric(key, value)
                        else:
                            for key, value in metrics:
                                st.metric(key, value)
                    
                    with sub_tab2:
                        st.json(result)
    
    with tab2:
        st.markdown("### 📁 All Generated Files")
        
        if os.path.exists(output_dir):
            all_files = []
            total_size = 0
            
            for root, dirs, filenames in os.walk(output_dir):
                for filename in filenames:
                    filepath = os.path.join(root, filename)
                    rel_path = os.path.relpath(filepath, output_dir)
                    file_size = os.path.getsize(filepath)
                    total_size += file_size
                    
                    # Determine which analysis this file belongs to
                    analysis_folder = rel_path.split(os.sep)[0] if os.sep in rel_path else "Root"
                    
                    all_files.append({
                        "📁 Analysis": analysis_folder,
                        "📄 File": rel_path,
                        "📏 Size": f"{file_size / 1024:.1f} KB",
                        "📂 Type": os.path.splitext(filename)[1] or "Directory",
                        "🕒 Modified": datetime.fromtimestamp(os.path.getmtime(filepath)).strftime('%Y-%m-%d %H:%M:%S')
                    })
            
            if all_files:
                st.info(f"📊 **Total: {len(all_files)} files across all analyses, {total_size / (1024*1024):.1f} MB**")
                
                # Display files table with filtering
                files_df = pd.DataFrame(all_files)
                
                # Filter options
                col1, col2 = st.columns(2)
                with col1:
                    analysis_filter = st.selectbox("Filter by Analysis:", 
                                                 ["All"] + list(files_df["📁 Analysis"].unique()))
                with col2:
                    type_filter = st.selectbox("Filter by File Type:", 
                                             ["All"] + list(files_df["📂 Type"].unique()))
                
                # Apply filters
                filtered_df = files_df.copy()
                if analysis_filter != "All":
                    filtered_df = filtered_df[filtered_df["📁 Analysis"] == analysis_filter]
                if type_filter != "All":
                    filtered_df = filtered_df[filtered_df["📂 Type"] == type_filter]
                
                st.dataframe(filtered_df, use_container_width=True)
            else:
                st.warning("No files found")
    
    with tab3:
        st.markdown("### ⬇️ Bulk Download Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📦 Complete Package")
            
            # Create comprehensive zip file
            import zipfile
            zip_path = os.path.join(output_dir, 'all_analysis_results.zip')
            
            try:
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, dirs, files in os.walk(output_dir):
                        for file in files:
                            if file != 'all_analysis_results.zip':
                                file_path = os.path.join(root, file)
                                arcname = os.path.relpath(file_path, output_dir)
                                zipf.write(file_path, arcname)
                
                zip_size = os.path.getsize(zip_path) / (1024*1024)
                st.info(f"📊 Complete package: {zip_size:.1f} MB")
                
                with open(zip_path, 'rb') as f:
                    st.download_button(
                        label="📦 **Download All Results (ZIP)**",
                        data=f.read(),
                        file_name=f"multi_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                        mime="application/zip",
                        use_container_width=True,
                        type="primary"
                    )
            except Exception as e:
                st.error(f"Error creating download package: {e}")
        
        with col2:
            st.markdown("#### 📊 Results Summary")
            
            # Create a summary report
            summary_report = f"""
Multi-Analysis Results Summary
==============================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Analyses: {total_analyses}
Successful: {successful_analyses}
Failed: {failed_analyses}
Success Rate: {(successful_analyses/total_analyses)*100:.1f}%

Analysis Details:
"""
            
            for analysis_name, result in all_results.items():
                summary_report += f"\n{analysis_name}:\n"
                if "error" in result:
                    summary_report += f"  Status: FAILED - {result['error']}\n"
                else:
                    summary_report += f"  Status: SUCCESS\n"
                    for key, value in list(result.items())[:3]:  # First 3 metrics
                        summary_report += f"  {key}: {value}\n"
            
            st.download_button(
                label="📄 **Download Summary Report**",
                data=summary_report,
                file_name=f"analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
    
    # Final status and celebration
    if successful_analyses == total_analyses and total_analyses > 0:
        st.balloons()
        st.success(f"🎉 All {total_analyses} analyses completed successfully!")
    elif successful_analyses > 0:
        st.success(f"✅ {successful_analyses} out of {total_analyses} analyses completed successfully.")
        if failed_analyses > 0:
            st.warning(f"⚠️ {failed_analyses} analyses failed. Check individual results for details.")
    else:
        st.error("❌ All analyses failed. Please check your input files and parameters.")

def run_multiple_analyses_simple(selected_analyses, fasta_path, json_path, jsonl_path, base_temp_dir):
    """Run multiple analyses with default parameters (simplified)"""
    
    st.session_state.analysis_status = "Running"
    st.session_state.analysis_progress = 0.0
    
    total_analyses = len(selected_analyses)
    all_results = {}
    
    st.info(f"🚀 Starting {total_analyses} analyses with default settings...")
    
    # Create progress container
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, analysis_name in enumerate(selected_analyses):
        try:
            # Update progress
            progress = (i + 1) / total_analyses
            progress_bar.progress(progress)
            status_text.text(f"Running {i+1}/{total_analyses}: {analysis_name}")
            
            # Create separate output directory for each analysis
            analysis_output_dir = os.path.join(base_temp_dir, f"analysis_{i+1}_{analysis_name.replace(' ', '_')}")
            os.makedirs(analysis_output_dir, exist_ok=True)
            
            # Use default parameters for each analysis
            default_params = get_default_parameters(analysis_name)
            
            # Run the analysis
            with st.spinner(f"Running {analysis_name}..."):
                if analysis_name == "mRNA Simulation Platform":
                    result = run_mrna_simulation(fasta_path, json_path, jsonl_path, default_params, analysis_output_dir)
                elif analysis_name == "FastQC Quality Control":
                    result = run_fastqc_analysis({}, default_params, analysis_output_dir)
                elif analysis_name == "Gene Sequence Optimization":
                    result = run_gene_optimization(fasta_path, json_path, jsonl_path, default_params, analysis_output_dir)
                elif analysis_name == "mRNA Vaccine Design":
                    result = run_vaccine_design(fasta_path, json_path, jsonl_path, default_params, analysis_output_dir)
                elif analysis_name == "mRNA Folding Optimization":
                    result = run_folding_optimization(default_params, analysis_output_dir)
                elif analysis_name == "MHCflurry Integration":
                    result = run_mhcflurry_analysis(fasta_path, json_path, jsonl_path, analysis_output_dir)
                elif analysis_name == "Nanopore Sequencing Analysis":
                    result = run_nanopore_analysis(fasta_path, json_path, jsonl_path, default_params, analysis_output_dir)
                elif analysis_name == "Illumina Sequencing Analysis":
                    result = run_illumina_analysis(fasta_path, {}, default_params, analysis_output_dir)
                elif analysis_name == "PacBio SMRT Analysis":
                    result = run_pacbio_analysis(default_params, analysis_output_dir)
                elif analysis_name == "Spatial Transcriptomics":
                    result = run_spatial_analysis(default_params, analysis_output_dir)
                else:
                    result = {"error": f"Unknown analysis: {analysis_name}"}
            
            all_results[analysis_name] = result
            
            # Show individual result
            if "error" not in result:
                st.success(f"✅ {analysis_name} completed successfully!")
            else:
                st.error(f"❌ {analysis_name} failed: {result['error']}")
            
        except Exception as e:
            st.error(f"❌ {analysis_name} failed with exception: {str(e)}")
            all_results[analysis_name] = {"error": str(e)}
    
    # Update final status
    progress_bar.progress(1.0)
    status_text.text("All analyses completed!")
    
    st.session_state.analysis_status = "Complete"
    st.session_state.analysis_progress = 1.0
    st.session_state.analysis_results = all_results
    
    # Display combined results
    display_multiple_results(all_results, base_temp_dir)

def get_default_parameters(analysis_name):
    """Get default parameters for each analysis (no user input required)"""
    defaults = {
        'mRNA Simulation Platform': {
            'create_sample': True,
            'simulation_time': 100,
            'num_sequences': 3
        },
        'FastQC Quality Control': {
            'create_sample': True,
            'num_samples': 3
        },
        'Gene Sequence Optimization': {
            'create_sample': True,
            'target_gc': 50,
            'max_sequences': 5
        },
        'mRNA Vaccine Design': {
            'max_sequences': 3,
            'populations': ['European', 'Asian']
        },
        'mRNA Folding Optimization': {
            'create_sample': True
        },
        'MHCflurry Integration': {
            'create_sample': True
        },
        'Nanopore Sequencing Analysis': {
            'create_sample': True,
            'num_reads': 5
        },
        'Illumina Sequencing Analysis': {
            'create_sample': True,
            'num_variants': 1000
        },
        'PacBio SMRT Analysis': {
            'create_sample': True,
            'num_reads': 1000,
            'min_accuracy': 0.99
        },
        'Spatial Transcriptomics': {
            'create_sample': True,
            'n_spots': 2000,
            'n_genes': 500
        }
    }
    
    return defaults.get(analysis_name, {'create_sample': True})
    """Get analysis-specific parameters for multi-analysis mode"""
    params = {}
    
    if analysis_name == "mRNA Simulation Platform":
        params['simulation_time'] = st.slider(f"Simulation Time (steps)", 50, 500, 100, key=f"sim_time_{analysis_name}")
        params['num_sequences'] = st.slider(f"Number of sequences", 1, 10, 3, key=f"num_seq_{analysis_name}")
    
    elif analysis_name == "FastQC Quality Control":
        params['num_samples'] = st.slider(f"Number of sample files", 1, 10, 3, key=f"num_samples_{analysis_name}")
    
    elif analysis_name == "Gene Sequence Optimization":
        params['target_gc'] = st.slider(f"Target GC Content (%)", 30, 70, 50, key=f"target_gc_{analysis_name}")
        params['max_sequences'] = st.slider(f"Max sequences to process", 1, 20, 5, key=f"max_seq_{analysis_name}")
    
    elif analysis_name == "mRNA Vaccine Design":
        params['max_sequences'] = st.slider(f"Max sequences to analyze", 1, 10, 3, key=f"max_seq_vaccine_{analysis_name}")
        params['populations'] = st.multiselect(
            f"Target populations",
            ['European', 'Asian', 'African', 'Hispanic'],
            default=['European', 'Asian'],
            key=f"populations_{analysis_name}"
        )
    
    elif analysis_name == "Nanopore Sequencing Analysis":
        params['num_reads'] = st.slider(f"Number of reads per file", 1, 20, 5, key=f"num_reads_nano_{analysis_name}")
    
    elif analysis_name == "Illumina Sequencing Analysis":
        params['num_variants'] = st.slider(f"Number of variants", 100, 5000, 1000, key=f"num_variants_{analysis_name}")
    
    elif analysis_name == "PacBio SMRT Analysis":
        params['num_reads'] = st.slider(f"Number of HiFi reads", 100, 5000, 1000, key=f"num_reads_pacbio_{analysis_name}")
        params['min_accuracy'] = st.slider(f"Minimum accuracy", 0.99, 0.999, 0.99, 0.001, key=f"min_acc_{analysis_name}")
    
    elif analysis_name == "Spatial Transcriptomics":
        params['n_spots'] = st.slider(f"Number of spots", 500, 10000, 2000, key=f"n_spots_{analysis_name}")
        params['n_genes'] = st.slider(f"Number of genes", 100, 2000, 500, key=f"n_genes_{analysis_name}")
    
    return params

def run_multiple_analyses(selected_analyses, fasta_path, json_path, jsonl_path, combined_params, base_temp_dir):
    """Run multiple analyses sequentially"""
    
    st.session_state.analysis_status = "Running"
    st.session_state.analysis_progress = 0.0
    
    total_analyses = len(selected_analyses)
    all_results = {}
    
    st.info(f"🚀 Starting {total_analyses} analyses...")
    
    # Create progress container
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, analysis_name in enumerate(selected_analyses):
        try:
            # Update progress
            progress = (i + 1) / total_analyses
            progress_bar.progress(progress)
            status_text.text(f"Running {i+1}/{total_analyses}: {analysis_name}")
            
            # Create separate output directory for each analysis
            analysis_output_dir = os.path.join(base_temp_dir, f"analysis_{i+1}_{analysis_name.replace(' ', '_')}")
            os.makedirs(analysis_output_dir, exist_ok=True)
            
            # Get parameters for this analysis
            global_params = combined_params['global']
            specific_params = combined_params['specific'].get(analysis_name, {})
            
            # Merge parameters
            params = {**global_params, **specific_params}
            
            # Run the analysis
            with st.spinner(f"Running {analysis_name}..."):
                if analysis_name == "mRNA Simulation Platform":
                    result = run_mrna_simulation(fasta_path, json_path, jsonl_path, params, analysis_output_dir)
                elif analysis_name == "FastQC Quality Control":
                    result = run_fastqc_analysis({}, params, analysis_output_dir)
                elif analysis_name == "Gene Sequence Optimization":
                    result = run_gene_optimization(fasta_path, json_path, jsonl_path, params, analysis_output_dir)
                elif analysis_name == "mRNA Vaccine Design":
                    result = run_vaccine_design(fasta_path, json_path, jsonl_path, params, analysis_output_dir)
                elif analysis_name == "mRNA Folding Optimization":
                    result = run_folding_optimization(params, analysis_output_dir)
                elif analysis_name == "MHCflurry Integration":
                    result = run_mhcflurry_analysis(fasta_path, json_path, jsonl_path, analysis_output_dir)
                elif analysis_name == "Nanopore Sequencing Analysis":
                    result = run_nanopore_analysis(fasta_path, json_path, jsonl_path, params, analysis_output_dir)
                elif analysis_name == "Illumina Sequencing Analysis":
                    result = run_illumina_analysis(fasta_path, {}, params, analysis_output_dir)
                elif analysis_name == "PacBio SMRT Analysis":
                    result = run_pacbio_analysis(params, analysis_output_dir)
                elif analysis_name == "Spatial Transcriptomics":
                    result = run_spatial_analysis(params, analysis_output_dir)
                else:
                    result = {"error": f"Unknown analysis: {analysis_name}"}
            
            all_results[analysis_name] = result
            
            # Show individual result
            if "error" not in result:
                st.success(f"✅ {analysis_name} completed successfully!")
            else:
                st.error(f"❌ {analysis_name} failed: {result['error']}")
            
        except Exception as e:
            st.error(f"❌ {analysis_name} failed with exception: {str(e)}")
            all_results[analysis_name] = {"error": str(e)}
    
    # Update final status
    progress_bar.progress(1.0)
    status_text.text("All analyses completed!")
    
    st.session_state.analysis_status = "Complete"
    st.session_state.analysis_progress = 1.0
    st.session_state.analysis_results = all_results
    
    # Display combined results
    display_multiple_results(all_results, base_temp_dir)
    """Get analysis-specific parameters"""
    params = {}
    
    if analysis_name == "mRNA Simulation Platform":
        params['simulation_time'] = st.slider("Simulation Time (steps)", 50, 500, 100)
        params['create_sample'] = st.checkbox("Create sample sequences", value=True)
        params['num_sequences'] = st.slider("Number of sequences", 1, 10, 3)
    
    elif analysis_name == "FastQC Quality Control":
        params['create_sample'] = st.checkbox("Create sample FASTQ files", value=True)
        params['num_samples'] = st.slider("Number of sample files", 1, 10, 3)
    
    elif analysis_name == "Gene Sequence Optimization":
        params['target_gc'] = st.slider("Target GC Content (%)", 30, 70, 50)
        params['create_sample'] = st.checkbox("Create sample sequences", value=True)
        params['max_sequences'] = st.slider("Max sequences to process", 1, 20, 5)
    
    elif analysis_name == "mRNA Vaccine Design":
        params['max_sequences'] = st.slider("Max sequences to analyze", 1, 10, 3)
        params['populations'] = st.multiselect(
            "Target populations",
            ['European', 'Asian', 'African', 'Hispanic'],
            default=['European', 'Asian']
        )
    
    elif analysis_name == "Nanopore Sequencing Analysis":
        params['create_sample'] = st.checkbox("Create sample FAST5 files", value=True)
        params['num_reads'] = st.slider("Number of reads per file", 1, 20, 5)
    
    elif analysis_name == "Illumina Sequencing Analysis":
        params['create_sample'] = st.checkbox("Create sample VCF", value=True)
        params['num_variants'] = st.slider("Number of variants", 100, 5000, 1000)
    
    elif analysis_name == "PacBio SMRT Analysis":
        params['create_sample'] = st.checkbox("Create sample HiFi reads", value=True)
        params['num_reads'] = st.slider("Number of HiFi reads", 100, 5000, 1000)
        params['min_accuracy'] = st.slider("Minimum accuracy", 0.99, 0.999, 0.99, 0.001)
    
    elif analysis_name == "Spatial Transcriptomics":
        params['create_sample'] = st.checkbox("Create sample spatial data", value=True)
        params['n_spots'] = st.slider("Number of spots", 500, 10000, 2000)
        params['n_genes'] = st.slider("Number of genes", 100, 2000, 500)
    
    return params

def run_analysis(analysis_name, fasta_path, json_path, jsonl_path, additional_files, params, temp_dir):
    """Run the selected analysis pipeline"""
    
    st.session_state.analysis_status = "Running"
    st.session_state.analysis_progress = 0.1
    
    try:
        with st.spinner(f"Running {analysis_name}..."):
            output_dir = os.path.join(temp_dir, 'output')
            os.makedirs(output_dir, exist_ok=True)
            
            results = {}
            
            if analysis_name == "mRNA Simulation Platform":
                results = run_mrna_simulation(fasta_path, json_path, jsonl_path, params, output_dir)
            
            elif analysis_name == "FastQC Quality Control":
                results = run_fastqc_analysis(additional_files, params, output_dir)
            
            elif analysis_name == "Gene Sequence Optimization":
                results = run_gene_optimization(fasta_path, json_path, jsonl_path, params, output_dir)
            
            elif analysis_name == "mRNA Vaccine Design":
                results = run_vaccine_design(fasta_path, json_path, jsonl_path, params, output_dir)
            
            elif analysis_name == "mRNA Folding Optimization":
                results = run_folding_optimization(params, output_dir)
            
            elif analysis_name == "MHCflurry Integration":
                results = run_mhcflurry_analysis(fasta_path, json_path, jsonl_path, output_dir)
            
            elif analysis_name == "Nanopore Sequencing Analysis":
                results = run_nanopore_analysis(fasta_path, json_path, jsonl_path, params, output_dir)
            
            elif analysis_name == "Illumina Sequencing Analysis":
                results = run_illumina_analysis(fasta_path, additional_files, params, output_dir)
            
            elif analysis_name == "PacBio SMRT Analysis":
                results = run_pacbio_analysis(params, output_dir)
            
            elif analysis_name == "Spatial Transcriptomics":
                results = run_spatial_analysis(params, output_dir)
            
            st.session_state.analysis_progress = 1.0
            st.session_state.analysis_status = "Complete"
            st.session_state.analysis_results = results
            
            # Display results
            display_results(results, output_dir)
            
    except Exception as e:
        st.session_state.analysis_status = "Error"
        st.error(f"Analysis failed: {str(e)}")
        st.exception(e)

def run_mrna_simulation(fasta_path, json_path, jsonl_path, params, output_dir):
    """Run mRNA simulation analysis"""
    if not MODULES_AVAILABLE:
        return simulate_results("mRNA Simulation")
    
    try:
        # Create platform instance
        platform = biosimai.mRNASimulationPlatform(output_dir=output_dir)
        
        # Get sequences
        if params.get('create_sample', True):
            sequences = platform.create_sample_mrna_sequences(num_sequences=params.get('num_sequences', 3))
        else:
            sequences = platform.load_mrna_from_fasta(fasta_path)
        
        if not sequences:
            return {"error": "No sequences found"}
        
        # Run simulations
        all_results = []
        for seq_info in sequences:
            result = platform.run_comprehensive_simulation(seq_info['sequence'], seq_info['id'])
            if result:
                all_results.append(result)
        
        return {
            "Total Sequences": len(sequences),
            "Successful Simulations": len(all_results),
            "Output Directory": output_dir,
            "Simulation Time": params.get('simulation_time', 100)
        }
    
    except Exception as e:
        return {"error": str(e)}

def run_fastqc_analysis(additional_files, params, output_dir):
    """Run FastQC quality control analysis"""
    if not MODULES_AVAILABLE:
        return simulate_results("FastQC")
    
    try:
        processor = fastqc.FastQCProcessor(output_dir=output_dir)
        
        fastq_files = []
        
        if params.get('create_sample', True):
            # Create sample files
            sample_dir = os.path.join(output_dir, 'sample_fastq')
            os.makedirs(sample_dir, exist_ok=True)
            
            for i in range(params.get('num_samples', 3)):
                sample_file = os.path.join(sample_dir, f'sample_{i}.fastq')
                processor.create_sample_fastq(sample_file, num_reads=np.random.randint(5000, 15000))
                fastq_files.append(sample_file)
        
        # Add uploaded FASTQ files
        if "FASTQ files for QC" in additional_files:
            fastq_files.extend(additional_files["FASTQ files for QC"])
        
        # Process files
        processed_samples = 0
        for fastq_file in fastq_files:
            result = processor.process_sample(fastq_file)
            if result:
                processed_samples += 1
        
        if processed_samples > 0:
            processor.generate_batch_report()
        
        return {
            "FASTQ Files Processed": processed_samples,
            "Total Files": len(fastq_files),
            "Output Directory": output_dir
        }
    
    except Exception as e:
        return {"error": str(e)}

def run_gene_optimization(fasta_path, json_path, jsonl_path, params, output_dir):
    """Run gene optimization analysis"""
    if not MODULES_AVAILABLE:
        return simulate_results("Gene Optimization")
    
    try:
        optimizer = gene_optimizer.GeneOptimizer(
            reference_fasta=fasta_path,
            dataset_json=json_path,
            assembly_jsonl=jsonl_path,
            output_dir=output_dir
        )
        
        # Get sequences
        if params.get('create_sample', True):
            sequences = optimizer.create_sample_sequences()
        else:
            sequences = optimizer.extract_cds_from_fasta(max_sequences=params.get('max_sequences', 5))
        
        if not sequences:
            return {"error": "No sequences found"}
        
        # Run optimization
        results = optimizer.batch_optimize_sequences(
            sequences,
            target_gc=params.get('target_gc', 50)
        )
        
        # Generate outputs
        optimizer.generate_optimization_plots(results)
        optimizer.save_optimized_sequences(results)
        optimizer.save_results(results)
        
        successful = sum(1 for r in results if r.get('optimization_successful', False))
        
        return {
            "Total Sequences": len(sequences),
            "Successful Optimizations": successful,
            "Target GC Content": f"{params.get('target_gc', 50)}%",
            "Output Directory": output_dir
        }
    
    except Exception as e:
        return {"error": str(e)}

def run_vaccine_design(fasta_path, json_path, jsonl_path, params, output_dir):
    """Run mRNA vaccine design analysis"""
    if not MODULES_AVAILABLE:
        return simulate_results("Vaccine Design")
    
    try:
        designer = iedb.AdvancedmRNAVaccineDesigner(fasta_path, json_path, jsonl_path)
        
        results = designer.run_comprehensive_pipeline(
            max_sequences=params.get('max_sequences', 3),
            output_dir=output_dir
        )
        
        if 'error' in results:
            return {"error": results['error']}
        
        # Generate report
        report = designer.generate_comprehensive_report(results, f"{output_dir}/vaccine_report.txt")
        
        return {
            "Sequences Processed": results['summary']['sequences_processed'],
            "Peptides Generated": results['summary']['peptides_generated'],
            "MHC Predictions": results['summary']['mhc_predictions'],
            "Construct Epitopes": results['summary']['construct_epitopes'],
            "Output Directory": output_dir
        }
    
    except Exception as e:
        return {"error": str(e)}

def run_folding_optimization(params, output_dir):
    """Run mRNA folding optimization"""
    if not MODULES_AVAILABLE:
        return simulate_results("Folding Optimization")
    
    try:
        optimizer = linear_design.mRNAFoldingOptimizer()
        
        # Example protein sequence
        example_protein = "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT"
        
        # Run design
        design_result = optimizer.design_complete_mrna(
            example_protein[:200],  # Use shorter sequence for demo
            include_utrs=True,
            kozak_optimization=True
        )
        
        if 'error' in design_result:
            return {"error": design_result['error']}
        
        # Generate report
        report = optimizer.create_optimization_report(design_result, f"{output_dir}/folding_report.txt")
        
        # Generate visualizations
        optimizer.visualize_optimization_results(design_result, output_dir)
        
        summary = design_result['design_summary']
        
        return {
            "mRNA Length": f"{summary['total_length']} nt",
            "GC Content": summary['gc_content'],
            "Translation Efficiency": summary['predicted_translation_efficiency'],
            "Manufacturing Feasibility": summary['manufacturing_feasibility'],
            "Output Directory": output_dir
        }
    
    except Exception as e:
        return {"error": str(e)}

def run_mhcflurry_analysis(fasta_path, json_path, jsonl_path, output_dir):
    """Run MHCflurry epitope analysis"""
    if not MODULES_AVAILABLE:
        return simulate_results("MHCflurry")
    
    try:
        predictor = mhcflurry_download.IEDBmRNAPredictor(fasta_path, json_path, jsonl_path)
        
        results = predictor.run_epitope_pipeline(output_dir=output_dir)
        
        if 'error' in results:
            return {"error": results['error']}
        
        # Generate report
        report = predictor.generate_report(results, f"{output_dir}/iedb_report.txt")
        
        return {
            "Sequences Processed": results['summary']['total_sequences'],
            "Peptides Generated": results['summary']['total_peptides'],
            "IEDB Predictions": results['summary']['iedb_predictions'],
            "Output Directory": output_dir
        }
    
    except Exception as e:
        return {"error": str(e)}

def run_nanopore_analysis(fasta_path, json_path, jsonl_path, params, output_dir):
    """Run Nanopore sequencing analysis"""
    if not MODULES_AVAILABLE:
        return simulate_results("Nanopore")
    
    try:
        processor = nanopore.NanoporeProcessor(fasta_path, json_path, jsonl_path)
        
        if params.get('create_sample', True):
            # Create sample FAST5 files
            sample_dir = os.path.join(output_dir, 'sample_fast5')
            os.makedirs(sample_dir, exist_ok=True)
            
            fast5_files = []
            for i in range(3):
                sample_file = os.path.join(sample_dir, f'sample_{i}.fast5')
                nanopore.create_sample_fast5(sample_file, num_reads=params.get('num_reads', 5))
                fast5_files.append(sample_file)
        
        # Process files
        all_processed_reads = []
        for fast5_file in fast5_files:
            reads_data = processor.read_fast5_file(fast5_file)
            if reads_data:
                processed_reads = processor.process_reads(reads_data)
                all_processed_reads.extend(processed_reads)
        
        # Save results
        if all_processed_reads:
            processor.save_results(all_processed_reads, output_dir)
        
        return {
            "FAST5 Files Processed": len(fast5_files),
            "Total Reads": len(all_processed_reads),
            "Output Directory": output_dir
        }
    
    except Exception as e:
        return {"error": str(e)}

def run_illumina_analysis(fasta_path, additional_files, params, output_dir):
    """Run Illumina sequencing analysis"""
    if not MODULES_AVAILABLE:
        return simulate_results("Illumina")
    
    try:
        processor = sequencing.IlluminaProcessor(reference_fasta=fasta_path)
        
        # Load metrics
        processor.load_interop_metrics()
        
        # Handle VCF file
        vcf_file = None
        if params.get('create_sample', True):
            vcf_file = os.path.join(output_dir, 'sample_variants.vcf')
            processor.create_sample_vcf(vcf_file, num_variants=params.get('num_variants', 1000))
        elif "VCF files" in additional_files:
            vcf_file = additional_files["VCF files"]
        
        if not vcf_file:
            return {"error": "No VCF file available"}
        
        # Load and analyze VCF data
        variants = processor.load_vcf_data(vcf_file)
        if not variants:
            return {"error": "Failed to load VCF data"}
        
        # Perform analyses
        snp_analysis = processor.detect_snps()
        indel_analysis = processor.detect_indels()
        af_analysis = processor.analyze_allele_frequencies()
        disease_variants = processor.identify_disease_linked_mutations()
        repeat_expansions = processor.detect_repeat_expansions()
        
        # Save results
        processor.save_results(
            output_dir, snp_analysis, indel_analysis,
            af_analysis, disease_variants, repeat_expansions
        )
        
        return {
            "Total Variants": len(processor.variants['variants/POS']),
            "SNPs Detected": snp_analysis['total_snps'] if snp_analysis else 0,
            "Indels Detected": indel_analysis['total_indels'] if indel_analysis else 0,
            "Disease Genes": len(disease_variants) if disease_variants else 0,
            "Output Directory": output_dir
        }
    
    except Exception as e:
        return {"error": str(e)}

def run_pacbio_analysis(params, output_dir):
    """Run PacBio SMRT analysis"""
    if not MODULES_AVAILABLE:
        return simulate_results("PacBio")
    
    try:
        processor = smrt.PacBioSMRTProcessor(output_dir=output_dir)
        
        # Create sample HiFi reads
        if params.get('create_sample', True):
            processor.create_sample_hifi_reads(
                num_reads=params.get('num_reads', 1000)
            )
        
        if not processor.hifi_reads:
            return {"error": "No HiFi reads available"}
        
        # Filter reads by quality
        min_accuracy = params.get('min_accuracy', 0.99)
        original_count = len(processor.hifi_reads)
        processor.hifi_reads = [
            read for read in processor.hifi_reads 
            if read['accuracy'] >= min_accuracy
        ]
        filtered_count = len(processor.hifi_reads)
        
        # Perform analyses
        variants = processor.detect_variants()
        transcripts = processor.discover_transcripts()
        
        # Generate plots
        plot_files = processor.generate_analysis_plots()
        
        # Save results
        processor.save_results()
        
        return {
            "HiFi Reads": f"{filtered_count} (filtered from {original_count})",
            "Variants Detected": len(variants),
            "Transcripts Discovered": len(transcripts),
            "Plots Generated": len(plot_files),
            "Output Directory": output_dir
        }
    
    except Exception as e:
        return {"error": str(e)}

def run_spatial_analysis(params, output_dir):
    """Run spatial transcriptomics analysis"""
    if not MODULES_AVAILABLE:
        return simulate_results("Spatial")
    
    try:
        processor = visium.SpatialTranscriptomicsProcessor(output_dir=output_dir)
        
        # Create sample spatial data
        if params.get('create_sample', True):
            adata = processor.create_sample_spatial_data(
                n_spots=params.get('n_spots', 2000),
                n_genes=params.get('n_genes', 500)
            )
        else:
            return {"error": "File upload not implemented yet"}
        
        if adata is None:
            return {"error": "Failed to create spatial data"}
        
        # Store original data
        processor.adata = adata
        
        # Preprocessing
        adata_processed = processor.preprocess_data(adata)
        if adata_processed is None:
            return {"error": "Preprocessing failed"}
        
        # Analyses
        adata_processed = processor.spatial_network_analysis(adata_processed)
        adata_processed = processor.dimensionality_reduction(adata_processed)
        adata_processed = processor.spatial_clustering(adata_processed)
        adata_processed = processor.gene_expression_patterns(adata_processed)
        
        # mRNA localization analysis
        localization_results = processor.mrna_localization_analysis(adata_processed)
        
        # Generate visualizations
        plot_files = processor.generate_spatial_plots(adata_processed)
        
        # Save results
        processor.save_results(adata_processed)
        
        return {
            "Spots Analyzed": processor.results['summary']['n_spots'],
            "Genes Analyzed": processor.results['summary']['n_genes'],
            "Spatial Clusters": processor.results.get('n_clusters', 'N/A'),
            "Plots Generated": len(plot_files),
            "Output Directory": output_dir
        }
    
    except Exception as e:
        return {"error": str(e)}

def simulate_results(analysis_type):
    """Simulate results when modules are not available"""
    return {
        "Status": "Simulated (modules not available)",
        "Analysis Type": analysis_type,
        "Note": "Install required dependencies to run actual analysis"
    }

def display_results(results, output_dir):
    """Display analysis results"""
    st.subheader("📊 Analysis Results")
    
    if "error" in results:
        st.error(f"Analysis failed: {results['error']}")
        return
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    metrics = list(results.items())
    for i, (key, value) in enumerate(metrics):
        if i % 3 == 0:
            col1.metric(key, value)
        elif i % 3 == 1:
            col2.metric(key, value)
        else:
            col3.metric(key, value)
    
    # Check for output files
    st.subheader("📁 Output Files")
    
    if os.path.exists(output_dir):
        files = []
        for root, dirs, filenames in os.walk(output_dir):
            for filename in filenames:
                filepath = os.path.join(root, filename)
                rel_path = os.path.relpath(filepath, output_dir)
                files.append({
                    "File": rel_path,
                    "Size": f"{os.path.getsize(filepath) / 1024:.1f} KB",
                    "Type": os.path.splitext(filename)[1] or "Directory"
                })
        
        if files:
            files_df = pd.DataFrame(files)
            st.dataframe(files_df, use_container_width=True)
            
            # Download section
            st.subheader("⬇️ Download Results")
            
            # Create zip file for download
            import zipfile
            zip_path = os.path.join(output_dir, 'results.zip')
            
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for root, dirs, filenames in os.walk(output_dir):
                    for filename in filenames:
                        if filename != 'results.zip':
                            filepath = os.path.join(root, filename)
                            arcname = os.path.relpath(filepath, output_dir)
                            zipf.write(filepath, arcname)
            
            with open(zip_path, 'rb') as f:
                st.download_button(
                    label="📦 Download All Results (ZIP)",
                    data=f.read(),
                    file_name=f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip"
                )
        else:
            st.info("No output files generated")
    else:
        st.warning("Output directory not found")

# Custom CSS for dark theme and better styling
st.markdown("""
<style>
    /* Dark theme for main app */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* Dark theme for sidebar */
    .css-1d391kg {
        background-color: #262730;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #262730;
        color: #fafafa;
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        background-color: #262730;
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 20px;
    }
    
    .stFileUploader label {
        color: #fafafa !important;
        font-weight: bold;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(76, 175, 80, 0.3);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div > select {
        background-color: #262730;
        color: #fafafa;
        border: 1px solid #4CAF50;
        border-radius: 5px;
    }
    
    /* Metric containers */
    .metric-container {
        background-color: #262730;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        border-left: 4px solid #4CAF50;
    }
    
    /* Info/Warning/Error boxes */
    .stAlert {
        background-color: #262730 !important;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
    }
    
    .stAlert > div {
        color: #fafafa !important;
    }
    
    /* Success message styling */
    .stSuccess {
        background-color: #1e4620 !important;
        color: #4CAF50 !important;
        border-left: 4px solid #4CAF50 !important;
    }
    
    /* Warning message styling */
    .stWarning {
        background-color: #4a3c1a !important;
        color: #ffc107 !important;
        border-left: 4px solid #ffc107 !important;
    }
    
    /* Error message styling */
    .stError {
        background-color: #4a1e1e !important;
        color: #f44336 !important;
        border-left: 4px solid #f44336 !important;
    }
    
    /* Info message styling */
    .stInfo {
        background-color: #1a3a4a !important;
        color: #2196F3 !important;
        border-left: 4px solid #2196F3 !important;
    }
    
    /* DataFrames */
    .stDataFrame {
        background-color: #262730;
        border-radius: 8px;
    }
    
    .stDataFrame table {
        background-color: #262730 !important;
        color: #fafafa !important;
    }
    
    .stDataFrame th {
        background-color: #4CAF50 !important;
        color: white !important;
    }
    
    /* Headers and text */
    h1, h2, h3, h4, h5, h6 {
        color: #fafafa !important;
    }
    
    /* Main content area styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Sidebar header styling */
    .sidebar .sidebar-content .sidebar-header {
        background-color: #4CAF50;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #262730;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #262730;
        color: #fafafa;
        border-radius: 4px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50 !important;
        color: white !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #262730;
        color: #fafafa !important;
        border: 1px solid #4CAF50;
        border-radius: 8px;
    }
    
    .streamlit-expanderContent {
        background-color: #1a1a1a;
        border: 1px solid #4CAF50;
        border-radius: 0 0 8px 8px;
    }
    
    /* Slider styling */
    .stSlider > div > div > div > div {
        background-color: #4CAF50;
    }
    
    /* Checkbox styling */
    .stCheckbox > label {
        color: #fafafa !important;
    }
    
    /* Download button special styling */
    .stDownloadButton > button {
        background-color: #2196F3;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: bold;
    }
    
    .stDownloadButton > button:hover {
        background-color: #1976D2;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(33, 150, 243, 0.3);
    }
    
    /* Footer styling */
    .footer {
        background-color: #262730;
        padding: 20px;
        border-radius: 8px;
        margin-top: 2rem;
        border-top: 2px solid #4CAF50;
    }
    
    /* Multiselect styling */
    .stMultiSelect > div > div > div {
        background-color: #262730;
        color: #fafafa;
    }
    
    /* Text input styling */
    .stTextInput > div > div > input {
        background-color: #262730;
        color: #fafafa;
        border: 1px solid #4CAF50;
        border-radius: 5px;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #262730;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #4CAF50;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #45a049;
    }
</style>
""", unsafe_allow_html=True)

# Footer with dark theme
st.markdown("---")
st.markdown("""
<div class='footer' style='text-align: center; color: #fafafa; padding: 20px; background-color: #262730; border-radius: 8px; margin-top: 2rem; border-top: 2px solid #4CAF50;'>
    <p>🧬 <strong>Bioinformatics Analysis Platform</strong> | Built with Streamlit</p>
    <p><em>Comprehensive pipeline for mRNA and genomic analysis</em></p>
    <p style='font-size: 0.8rem; color: #888;'>Dark mode enabled for better readability</p>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()