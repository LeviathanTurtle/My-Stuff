
# 
# William Wadsworth
# 
# Note that the PDFs tested are from Oxford's Learners Word Lists
# (https://www.oxfordlearnersdictionaries.com/us/wordlists/)
# 
# todo: verify new words
# 

from re import compile
from os import listdir, path, getcwd, remove
from PyPDF2 import PdfReader
from argparse import ArgumentParser

PATTERN = compile(r'^[^\s]+')

# pre-condition: 
# post-condition: 
def extract_words(pdf_path, temp_file):
    """Extracts words and writes them to an output file."""
    
    reader = PdfReader(pdf_path)
    with open(temp_file, 'w', encoding='utf-8') as outfile:
        for page in reader.pages:
            text = page.extract_text()
            if text: # ensure text was extracted from the page
                for line in text.split('\n'):
                    # match the word at the start of the line
                    match = PATTERN.match(line.strip())
                    if match:
                        # write the matched word to the temporary file
                        outfile.write(match.group(0) + '\n')

# pre-condition: 
# post-condition: 
def process_pdfs(directory):
    """Processes PDFs."""
    
    all_words_temp = "all_words_temp.txt"
    
    # remove temp file if it already exists
    if path.exists(all_words_temp):
        remove(all_words_temp)
    
    # for each PDF
    for file_name in listdir(directory):
        if file_name.endswith(".pdf"):
            pdf_path = path.join(directory, file_name)
            print(f"Processing {pdf_path}...")
            extract_words(pdf_path, all_words_temp)
    
    # sort and remove duplicates
    output_file = "sorted_unique_words.txt"
    # write to the final output file
    sort_and_remove_duplicates(all_words_temp, output_file)
    print(f"All unique words sorted and saved to {output_file}")
    
    # clean up temp file
    if path.exists(all_words_temp):
        remove(all_words_temp)

# pre-condition: 
# post-condition: 
def sort_and_remove_duplicates(temp_file, output_file):
    """Sorts the file alphabetically and removes duplicate words."""
    
    with open(temp_file, 'r', encoding='utf-8') as infile:
        words = infile.readlines()
    
    # strip any extra whitespace and remove duplicates by converting to a set and sort
    unique_words = sorted(set(word.strip() for word in words))
    
    # write to final output file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for word in unique_words:
            outfile.write(word + '\n')
            
# pre-condition: 
# post-condition: 
def add_words(new_words, output_file):
    """Adds words to the list file."""
    
    # read existing words from the final output file, if it exists
    if path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as infile:
            words = infile.readlines()
    else: words = []
    
    # add the new words to the existing words list
    words.extend(new_words)
    
    # strip any extra whitespace, remove duplicates, and sort
    unique_words = sorted(set(word.strip() for word in words))
    
    # write to output file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for word in unique_words:
            outfile.write(word + '\n')

    print(f"Words '{', '.join(new_words)}' added and file updated.")


if __name__ == "__main__":
    parser = ArgumentParser(description="Process PDF files and manage word list.")
    parser.add_argument('--add', nargs='*', help="Add new word(s) to the final word list.")
    parser.add_argument('--process', action='store_true', help="Process all PDFs in the current directory.")

    args = parser.parse_args()

    output_file = "self_words.txt"

    if args.process:
        current_directory = getcwd()
        process_pdfs(current_directory)
    
    if args.add:
        add_words(args.add, output_file)