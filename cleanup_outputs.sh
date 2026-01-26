#!/bin/bash
# Cleanup script to remove generated output files for re-processing
# This allows the workflow to regenerate files with updated formats

echo "Cleaning up output files..."

OUTPUT_DIR="outputs"

# Stage 2: Enriched users (needs regeneration for new purchase_history format)
rm -f "$OUTPUT_DIR/stage2_enriched_users.json"
rm -f "$OUTPUT_DIR/stage2_enriched_users_preview.json"
rm -f "$OUTPUT_DIR/stage2_users_summary.json"

# Stage 3: Cuisine profiles
rm -f "$OUTPUT_DIR/stage3_cuisine_profiles.json"
rm -f "$OUTPUT_DIR/stage3_cuisine_profiles_preview.json"

# Stage 4: Prompts
rm -f "$OUTPUT_DIR/stage4_prompts.json"
rm -f "$OUTPUT_DIR/stage4_prompts_preview.json"
rm -f "$OUTPUT_DIR/stage4_prompts_readable.txt"

# Stage 5: Predictions
rm -f "$OUTPUT_DIR/stage5_predictions.json"
rm -f "$OUTPUT_DIR/stage5_predictions_preview.json"
rm -f "$OUTPUT_DIR/stage5_predictions_summary.json"

# Stage 6: TopK evaluation
rm -f "$OUTPUT_DIR/stage6_topk_results.json"
rm -f "$OUTPUT_DIR/stage6_topk_samples.json"
rm -f "$OUTPUT_DIR/stage6_topk_samples_preview.json"
rm -f "$OUTPUT_DIR/stage6_topk_detailed.json"
rm -f "$OUTPUT_DIR/stage6_topk_detailed_preview.json"

# Stage 7: Rerank evaluation
rm -f "$OUTPUT_DIR/stage7_rerank_results.json"
rm -f "$OUTPUT_DIR/stage7_rerank_samples.json"
rm -f "$OUTPUT_DIR/stage7_rerank_samples_preview.json"
rm -f "$OUTPUT_DIR/stage7_rerank_detailed.json"
rm -f "$OUTPUT_DIR/stage7_rerank_detailed_preview.json"

echo "Cleanup complete!"
echo ""
echo "Files removed. Stage 1 data (merged_data.parquet) kept intact."
echo "Run the workflow to regenerate files with new formats."
