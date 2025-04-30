OUTPUT_DIR=output/
python -m src.relation_encoders.compute_features \
    --dataset nr3d \
    --output $OUTPUT_DIR \
    --label pred
