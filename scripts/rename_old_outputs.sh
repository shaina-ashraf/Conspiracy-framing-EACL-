#!/bin/bash
# Rename old output files (from experiments with supportive label) to avoid overwriting
# Files with "un-labelled" or "un_labelled" in the name are NOT renamed (as per user request)

set -euo pipefail

OUTPUTS_DIR="/home/ashrafs/projects/Conspiracy-Proj/outputs"
SUFFIX="_v1_with_supportive"

echo "========================================="
echo "Renaming old output files (seen data only)"
echo "Suffix: ${SUFFIX}"
echo "========================================="
echo ""

cd "$OUTPUTS_DIR"

# Find all CSV and JSONL files that:
# 1. Do NOT contain "un-labelled" or "un_labelled" in the name
# 2. Match the pattern for seen data annotations
# 3. Don't already have the suffix

renamed_count=0
skipped_count=0

# Process CSV files
for file in *.csv; do
    if [[ ! -f "$file" ]]; then
        continue
    fi
    
    # Skip if already renamed
    if [[ "$file" == *"${SUFFIX}"* ]]; then
        echo "⏭️  Skipping (already renamed): $file"
        ((skipped_count++))
        continue
    fi
    
    # Skip un-labelled files (unseen data)
    if [[ "$file" == *"un-labelled"* ]] || [[ "$file" == *"un_labelled"* ]]; then
        echo "⏭️  Skipping (un-labelled data): $file"
        ((skipped_count++))
        continue
    fi
    
    # Skip files in subdirectories (like human_review_batches/)
    if [[ "$file" == *"/"* ]]; then
        continue
    fi
    
    # Check if it's a seen data annotation file
    # Pattern: {model}-{prompt}_all_classifications*.csv or {model}-{prompt}_all_span_annotations*.csv
    if [[ "$file" == *"_all_classifications"* ]] || \
       [[ "$file" == *"_all_span_annotations"* ]] || \
       [[ "$file" == *"_annotation_failures"* ]]; then
        # Extract base name and extension
        base="${file%.*}"
        ext="${file##*.}"
        new_name="${base}${SUFFIX}.${ext}"
        
        if [[ -f "$new_name" ]]; then
            echo "⚠️  Warning: Target already exists, skipping: $file -> $new_name"
            ((skipped_count++))
        else
            mv "$file" "$new_name"
            echo "✅ Renamed: $file -> $new_name"
            ((renamed_count++))
        fi
    else
        echo "⏭️  Skipping (not an annotation file): $file"
        ((skipped_count++))
    fi
done

# Process JSONL files
for file in *.jsonl; do
    if [[ ! -f "$file" ]]; then
        continue
    fi
    
    # Skip if already renamed
    if [[ "$file" == *"${SUFFIX}"* ]]; then
        echo "⏭️  Skipping (already renamed): $file"
        ((skipped_count++))
        continue
    fi
    
    # Skip un-labelled files (unseen data)
    if [[ "$file" == *"un-labelled"* ]] || [[ "$file" == *"un_labelled"* ]]; then
        echo "⏭️  Skipping (un-labelled data): $file"
        ((skipped_count++))
        continue
    fi
    
    # Skip files in subdirectories
    if [[ "$file" == *"/"* ]]; then
        continue
    fi
    
    # Check if it's an annotation failure file
    if [[ "$file" == *"_annotation_failures"* ]]; then
        base="${file%.*}"
        ext="${file##*.}"
        new_name="${base}${SUFFIX}.${ext}"
        
        if [[ -f "$new_name" ]]; then
            echo "⚠️  Warning: Target already exists, skipping: $file -> $new_name"
            ((skipped_count++))
        else
            mv "$file" "$new_name"
            echo "✅ Renamed: $file -> $new_name"
            ((renamed_count++))
        fi
    else
        echo "⏭️  Skipping (not an annotation file): $file"
        ((skipped_count++))
    fi
done

echo ""
echo "========================================="
echo "Summary:"
echo "  ✅ Renamed: ${renamed_count} files"
echo "  ⏭️  Skipped: ${skipped_count} files"
echo "========================================="
echo ""
echo "Old files are now marked with suffix: ${SUFFIX}"
echo "You can safely run new experiments without overwriting old results."

