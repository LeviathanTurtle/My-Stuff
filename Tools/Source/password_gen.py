# 
# WILLIAM WADSWORTH
# Simple password generator
# 10.09.2024
# 

from random import sample, choice

# this should be the file containing the words you want to pull from
WORD_FILE = "self_words.txt"

def genPassword(
    length: int = 15,
    memorable: bool = False,
    num_words: int = 3,
    word_length: int = 7,
    random_chars: bool = False,
    special_char: str = "-"
) -> str:
    ALPHABET = "abcdefghijklmnopqrstuvwxyz"
    BIG_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    SYMBOLS = """`~!@#$%^&*()-_=+,<.>/?;:'"[{]}\|"""
    NUMBER = "0123456789"
    #bank = ALPHABET + BIG_ALPHABET + SYMBOLS + NUMBER
    
    LETTERS = ALPHABET + BIG_ALPHABET
    NON_LETTERS = SYMBOLS + NUMBER
    
    # NOTE: num_words, word_length, random_chars, and special_char only apply if you are generating
    # a memorable password
    if memorable:
        try:
            with open(WORD_FILE,'r') as file:
                words = [word.strip() for word in file if len(word.strip()) >= word_length]
        except IOError:
            pass
        except FileNotFoundError:
            pass
        
        if len(words) < num_words:
            raise ValueError("Not enough words of the required length in the file.")
        
        # Select num_words random words
        selected_words = sample(words, num_words)
        password = ""
        
        # Build password by capitalizing first letter, appending a number and special character
        for word in selected_words:
            password += word.capitalize() + choice(NUMBER)
            if random_chars: password += choice(SYMBOLS)
            else: password += special_char
        
        # remove last separating char before returning
        return password[:-1]
    else: 
        #return "".join(sample(bank,length))
        num_letters = round(length * 0.7)  # 70% of the total length should be letters
        num_non_letters = length - num_letters
        
        # Combine and shuffle to create the final password
        password_list = sample(LETTERS, num_letters) + sample(NON_LETTERS, num_non_letters)
        return "".join(sample(password_list,length))

def main():
    # default
    print(f"Default password gen: {genPassword()}\n")
    # length
    print(f"Password length of 7: {genPassword(length=7)}")
    print(f"Length of 25: {genPassword(length=25)}\n")
    # memorable
    print(f"Memorable password: {genPassword(memorable=True)}")
    print(f"Memorable with words of at least 3 letters: {genPassword(memorable=True,word_length=3)}")
    print(f"Memorable with 5 words: {genPassword(memorable=True,num_words=5)}")
    print(f"Memorable with random chars: {genPassword(memorable=True,random_chars=True)}")
    print(f"Memorable with specific separating char: {genPassword(memorable=True,special_char=',')}")
    # note that you should include commas in the password so that if/when your creds are dumped to
    # a CSV it will break everything :)
    

if __name__ == "__main__":
    main()

