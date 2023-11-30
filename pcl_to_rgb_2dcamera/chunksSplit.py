def split_array_indices(array_length, num_chunks):
    # Calculate the size of each chunk
    chunk_size = array_length // num_chunks
    remainder = array_length % num_chunks

    # Initialize start and end indices
    start_index = 0
    end_index = chunk_size + (1 if remainder > 0 else 0)

    # Iterate to find the start and end indices for each chunk
    indices_list = []
    for _ in range(num_chunks):
        indices_list.append((start_index, end_index))
        start_index = end_index
        end_index = start_index + chunk_size + (1 if remainder > 1 else 0)
        remainder -= 1
    return indices_list

if __name__ == '__main__':
    # Example usage for an array of length 24 and splitting into 5 chunks
    array_length = 24
    num_chunks = 5
    indices = split_array_indices(array_length, num_chunks)
    # Print the start and end indices for each chunk
    for start, end in indices:
        print(f"Chunk: Start Index = {start}, End Index = {end - 1}")