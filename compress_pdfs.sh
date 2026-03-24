#!/bin/bash

# PDF Compression Script
# This script compresses all PDF files in the slides folder using Ghostscript

SLIDES_DIR="/home/msaad/githubRepos/deep-learning-ICTS6361/slides"
ORIGINAL_SIZE=0
COMPRESSED_SIZE=0

echo "Starting PDF compression for files in: $SLIDES_DIR"
echo "=================================================="

# Function to get file size in bytes
get_file_size() {
    stat -c%s "$1"
}

# Function to format bytes to human readable
format_size() {
    local size=$1
    if [ $size -ge 1048576 ]; then
        echo "$(( size / 1048576 ))MB"
    elif [ $size -ge 1024 ]; then
        echo "$(( size / 1024 ))KB"
    else
        echo "${size}B"
    fi
}

# Process each PDF file
for pdf_file in "$SLIDES_DIR"/*.pdf; do
    if [ -f "$pdf_file" ]; then
        filename=$(basename "$pdf_file" .pdf)
        compressed_file="$SLIDES_DIR/${filename}_compressed.pdf"
        
        # Skip if compressed version already exists
        if [ -f "$compressed_file" ]; then
            echo "Skipping $filename - compressed version already exists"
            continue
        fi
        
        echo "Compressing: $filename.pdf"
        
        # Get original size
        original_size=$(get_file_size "$pdf_file")
        ORIGINAL_SIZE=$((ORIGINAL_SIZE + original_size))
        
        # Compress using Ghostscript with high quality settings
        gs -sDEVICE=pdfwrite \
           -dCompatibilityLevel=1.4 \
           -dPDFSETTINGS=/ebook \
           -dNOPAUSE \
           -dQUIET \
           -dBATCH \
           -sOutputFile="$compressed_file" \
           "$pdf_file"
        
        # Get compressed size
        if [ -f "$compressed_file" ]; then
            compressed_size=$(get_file_size "$compressed_file")
            COMPRESSED_SIZE=$((COMPRESSED_SIZE + compressed_size))
            
            # Calculate reduction
            reduction=$((original_size - compressed_size))
            reduction_percent=$(( (reduction * 100) / original_size ))
            
            echo "  Original: $(format_size $original_size)"
            echo "  Compressed: $(format_size $compressed_size)"
            echo "  Reduction: $(format_size $reduction) ($reduction_percent%)"
            echo ""
        else
            echo "  Error: Failed to create compressed file"
        fi
    fi
done

echo "=================================================="
echo "Compression Summary:"
echo "Total original size: $(format_size $ORIGINAL_SIZE)"
echo "Total compressed size: $(format_size $COMPRESSED_SIZE)"
echo "Total space saved: $(format_size $((ORIGINAL_SIZE - COMPRESSED_SIZE)))"

if [ $ORIGINAL_SIZE -gt 0 ]; then
    total_reduction_percent=$(( ((ORIGINAL_SIZE - COMPRESSED_SIZE) * 100) / ORIGINAL_SIZE ))
    echo "Total reduction: $total_reduction_percent%"
fi

echo "=================================================="
echo "Compression completed!"
