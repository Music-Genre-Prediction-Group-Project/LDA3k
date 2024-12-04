import csv

# Specify the input and output file paths
input_file = 'dataset/genly_dataset_3k.csv'
output_file = 'dataset/genly3k_transformed.csv'

# Open the input file for reading
with open(input_file, mode='r', newline='', encoding='utf-8') as infile:
    reader = csv.reader(infile)
    
    # Open the output file for writing
    with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        
        # Process each row in the input file
        for row in reader:
            if len(row) < 2:
                # Skip rows that don't have at least track_id and genre
                continue
            
            # Extract track_id and genre
            track_id = row[0]
            genre = row[1]
            
            # Combine the remaining elements into a single lyric_data string
            lyric_data = ','.join(row[2:])
            
            # Write the new row to the output file
            writer.writerow([track_id, genre, lyric_data])
