import pandas as pd
from collections import defaultdict

# Load word IDs from the file
def load_word_ids(file_path):
    with open(file_path, 'r') as f:
        word_ids = f.read().split(',')
    return word_ids

# Parse genre-lyrics-3k.csv data to calculate song occurrence
def parse_lyrics_data(file_path):
    word_song_counts = defaultdict(int)  # Word -> Count of songs it appears in
    total_songs = 0  # Total number of songs

    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            total_songs += 1
            words_in_song = set()  # Keep track of words that appear in the current song
            
            for word_count in parts[2:]:
                word_id, _ = word_count.split(':')
                word_id = int(word_id) - 1  # Adjust to 0-based index
                words_in_song.add(word_id)
            
            for word_id in words_in_song:
                word_song_counts[word_id] += 1

    return word_song_counts, total_songs

# Calculate the percentage of songs containing each word
def calculate_word_song_percentages(word_song_counts, total_songs):
    word_song_percentages = {word_id: (count / total_songs) * 100 for word_id, count in word_song_counts.items()}
    return word_song_percentages

def main():
    # File paths (adjust as needed)
    genre_lyrics_file = 'dataset/genly_dataset_3k.csv'
    word_ids_file = 'dataset/word-ids-index-1.txt'
    
    # Load word IDs
    word_ids = load_word_ids(word_ids_file)
    
    # Parse lyrics data to get word song counts
    word_song_counts, total_songs = parse_lyrics_data(genre_lyrics_file)
    
    # Calculate word song percentages
    word_song_percentages = calculate_word_song_percentages(word_song_counts, total_songs)
    
    # Sort words by their percentage in descending order
    sorted_word_percentages = sorted(word_song_percentages.items(), key=lambda x: x[1], reverse=True)
    
    # Display results
    print(f"Total number of songs: {total_songs}\n")
    print("Percentage of Songs Containing Each Word (sorted by popularity):")
    for word_id, percentage in sorted_word_percentages:
        print(f"{word_ids[word_id]}: {percentage:.2f}%")
        if percentage < 10:
            break

if __name__ == '__main__':
    main()
