# I downloaded a .txt file containing 100,000+ lines of words to use as a word bank for some of my
# programs. I noticed some had apostrophes in them, and wanted to take them out. This is what this
# script does

with open("words.txt", "r") as input_file:
    lines = input_file.readlines()
    
    filtered_lines = [line for line in lines if "'" not in line]
    
    with open("words-alpha.txt", "w") as output_file:
        output_file.writelines(filtered_lines)

