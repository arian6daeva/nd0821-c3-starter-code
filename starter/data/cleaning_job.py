'''
The following code reads a CSV file called "census.csv", removes spaces after commas in each line, and saves the cleaned lines to a new file "clean_census.csv".
'''

# Define the input and output filenames
input_file = "starter/data/census.csv"
output_file = "starter/data/clean_census.csv"

# Read the input file, clean each line, and store the cleaned lines in a list
with open(input_file) as f:
    cleaned_lines = [line.replace(", ", ",") for line in f]

# Write the cleaned lines to the output file
with open(output_file, "w") as f:
    f.writelines(cleaned_lines)
