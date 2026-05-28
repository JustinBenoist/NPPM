DATASET="scenes/dataset/"
ITER=512

# Find all the .xml files in the dataset directory
FILES=$(find $DATASET -type f -name "*.xml")

# Loop through each file and run the command
for FILE in $FILES; do
    # Extract the base name of the file (without the path and extension)
    BASENAME=$(basename "$FILE" .xml)

    # Construct the output file name
    OUTFILE="$DATASET/mask_$BASENAME.exr"
    
    # Run the command with the current file
    python3 mask.py --scene "$FILE"  --iter $ITER --outfile "$OUTFILE"
done