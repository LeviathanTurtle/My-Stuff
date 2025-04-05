
# 
# A collection of cipher classes detailed in http://practicalcryptography.com/ciphers/. Most of
# these have already been done in the pycipher module (https://github.com/jameslyons/pycipher), but
# I thought it beneficial to do my own implementations.
# WILLIAM WADSWORTH
# todo look at test cases from module page

from string import ascii_lowercase, ascii_uppercase
ALPHABET_LENGTH: int = len(ascii_lowercase)

from sys import stderr
from typing import Tuple, List, Optional
from Libraries.debug_logger import DebugLogger

TEST_STRING: str = "Hello World"


# Atbash Cipher
class AtbashCipher:
    # substitution cipher where the letters are reversed (A = Z, B = Y, ...)
    def encode(text: str) -> str:
        alphabet = ascii_lowercase
        reversed_alphabet = alphabet[::-1]
        
        translation_table = str.maketrans(alphabet+alphabet.upper(), reversed_alphabet+reversed_alphabet.upper())
        
        return text.translate(translation_table)
    
    def decode(text: str) -> str:
        # since the cipher is symmetric we can just re-encode the text to decode it
        return AtbashCipher.encode(text)
#print(f"Encoded: {AtbashCipher.encode(TEST_STRING)}")
#print(f"Decoded: {AtbashCipher.decode(AtbashCipher.encode(TEST_STRING))}")

# ROT13 Cipher
class ROT13Cipher:
    # substitution cipher with a unique key where the letters are offset by 13 places
    def encode(text: str) -> str:
        alphabet = ascii_lowercase
        shifted_alphabet = alphabet[13:] + alphabet[:13]
        
        translation_table = str.maketrans(alphabet+alphabet.upper(), shifted_alphabet+shifted_alphabet.upper())
        
        return text.translate(translation_table)
    
    def decode(text: str) -> str:
        # since the cipher is symmetric we can just re-encode the text to decode it
        return ROT13Cipher.encode(text)
#print(f"Encoded: {ROT13Cipher.encode(TEST_STRING)}")
#print(f"Decoded: {ROT13Cipher.decode(ROT13Cipher.encode(TEST_STRING))}")

# Caesar Cipher
class CaesarCipher:
    # substitution cipher where each letter is offset a certain number of places
    def encode(text: str, shift: int) -> str:
        alphabet = ascii_lowercase
        shifted_alphabet = alphabet[shift%26:] + alphabet[:shift%26]
        
        translation_table = str.maketrans(alphabet+alphabet.upper(), shifted_alphabet+shifted_alphabet.upper())
        
        return text.translate(translation_table)
    
    def decode(text: str, shift: int) -> str:
        # since the cipher is symmetric we can just re-encode the text to decode it
        return CaesarCipher.encode(text,-shift)
#print(f"Encoded: {CaesarCipher.encode(TEST_STRING,5)}")
#print(f"Decoded: {CaesarCipher.decode(CaesarCipher.encode(TEST_STRING,5),5)}")

# Affine Cipher
class AffineCipher:
    # substitution cipher 
    
    # Helper function to compute the greatest common divisor
    def gcd(a: int, b: int) -> int:
        while b: a, b = b, a % b
        return a
    
    # Function to calculate modular inverse
    def mod_inverse(a: int, m: int) -> int:
        for i in range(m):
            if (a * i) % m == 1: return i
        raise ValueError(f"No modular inverse for a = {a} under modulo {m}.")

    def encode(text: str, a: int, b: int) -> str:
        # Ensure that 'a' is coprime with 26 (alphabet length)
        m = 26
        if AffineCipher.gcd(a, m) != 1:
            raise ValueError(f"'a' must be coprime with {m}.")
        # valid numbers are: 1,3,5,7,9,11,15,17,19,21,23,25
        #if a not in [1,3,5,7,9,11,15,17,19,21,23,25]:
        #   raise ValueError(f"'a' must be coprime with {m}.")
        
        alphabet = ascii_lowercase
        result = []

        for char in text:
            if char.isalpha():
                # Convert character to number (0-25 for a-z)
                x = alphabet.index(char.lower())
                # Apply the affine transformation: (a * x + b) % m
                encrypted_char = (a * x + b) % m
                # Convert number back to character
                if char.islower(): result.append(alphabet[encrypted_char])
                else: result.append(alphabet[encrypted_char].upper())
            else: result.append(char)  # Non-alphabetic characters are unchanged

        return ''.join(result)
    
    def decode(text: str, a: int, b: int) -> str:
        # Ensure 'a' has a modular inverse under modulo 26
        m = 26
        a_inv = AffineCipher.mod_inverse(a, m)  # Modular inverse of a
        
        alphabet = ascii_lowercase
        result = []

        for char in text:
            if char.isalpha():
                # Convert character to number (0-25 for a-z)
                x = alphabet.index(char.lower())
                # Apply the affine decryption formula: a_inv * (x - b) % m
                decrypted_char = (a_inv * (x - b)) % m
                # Convert number back to character
                if char.islower(): result.append(alphabet[decrypted_char])
                else: result.append(alphabet[decrypted_char].upper())
            else: result.append(char)  # Non-alphabetic characters are unchanged

        return ''.join(result)
#print(f"Encoded: {AffineCipher.encode(TEST_STRING,15,26)}")
#print(f"Decoded: {AffineCipher.decode(AffineCipher.encode(TEST_STRING,15,26),15,26)}")

# Rail-fence Cipher
class RailFenceCipher:
    # transposition cipher where the key denotes the number of 'rails' (encrypted letters)
    def encode(text: str, key: int) -> str:
        if key <= 1:
            return text
        
        text = text.replace(" ", "")  # Remove spaces
        fence = ["" for _ in range(key)]
        rail = 0
        direction = 1  # 1 for down, -1 for up
        
        for char in text:
            fence[rail] += char
            rail += direction
            if rail == 0 or rail == key - 1:
                direction *= -1
        
        return "".join(fence)
    
    def decode(text: str, key: int) -> str:
        if key <= 1:
            return text
        
        pattern = [0] * len(text)
        rail = 0
        direction = 1
        
        for i in range(len(text)):
            pattern[i] = rail
            rail += direction
            if rail == 0 or rail == key - 1:
                direction *= -1
        
        rail_lengths = [pattern.count(r) for r in range(key)]
        fence = []
        index = 0
        for length in rail_lengths:
            fence.append(list(text[index:index+length]))
            index += length
        
        result = []
        rail = 0
        direction = 1
        
        for _ in range(len(text)):
            result.append(fence[rail].pop(0))
            rail += direction
            if rail == 0 or rail == key - 1:
                direction *= -1
        
        return "".join(result)
    
    # I don't think this is working quite right
    def construct_output(text: str, key: int) -> str:
        if key <= 1:
            return text
        
        text = text.replace(" ", "")  # Remove spaces
        fence = [['.' for _ in range(len(text))] for _ in range(key)]
        rail = 0
        direction = 1
        
        for i, char in enumerate(text):
            fence[rail][i] = char
            rail += direction
            if rail == 0 or rail == key - 1:
                direction *= -1
        
        return '\n'.join(' '.join(row) for row in fence)
#print(f"Encoded: {RailFenceCipher.encode(TEST_STRING,3)}")
#print(f"Output:\n{RailFenceCipher.construct_output(RailFenceCipher.encode(TEST_STRING,3),3)}")
#print(f"Decoded: {RailFenceCipher.decode(RailFenceCipher.encode(TEST_STRING,3),3)}")

# Baconian Cipher
class BaconianCipher:
    def encode(text: str) -> str:
        pass
    
    def decode(text: str) -> str:
        pass
#print(f"Encoded: {BaconianCipher.encode(TEST_STRING)}")
#print(f"Decoded: {BaconianCipher.decode(BaconianCipher.encode(TEST_STRING))}")

# Polybius Square Cipher
class PolybiusCipher:
    def encode(text: str) -> str:
        pass
    
    def decode(text: str) -> str:
        pass
#print(f"Encoded: {PolybiusCipher.encode(TEST_STRING)}")
#print(f"Decoded: {PolybiusCipher.decode(PolybiusCipher.encode(TEST_STRING))}")

# Simple Substitution Cipher
class SimpleSubstitutionCipher:
    def encode(text: str) -> str:
        pass
    
    def decode(text: str) -> str:
        pass
#print(f"Encoded: {SimpleSubstitutionCipher.encode(TEST_STRING)}")
#print(f"Decoded: {SimpleSubstitutionCipher.decode(SimpleSubstitutionCipher.encode(TEST_STRING))}")

# Codes and Nomenclators Cipher
class CodesAndNomenclatorsCipher:
    def encode(text: str) -> str:
        pass
    
    def decode(text: str) -> str:
        pass
#print(f"Encoded: {CodesAndNomenclatorsCipher.encode(TEST_STRING)}")
#print(f"Decoded: {CodesAndNomenclatorsCipher.decode(CodesAndNomenclatorsCipher.encode(TEST_STRING))}")

# Columnar Transposition Cipher
class ColumnarTranspositionCipher:
    def encode(text: str) -> str:
        pass
    
    def decode(text: str) -> str:
        pass
#print(f"Encoded: {ColumnarTranspositionCipher.encode(TEST_STRING)}")
#print(f"Decoded: {ColumnarTranspositionCipher.decode(ColumnarTranspositionCipher.encode(TEST_STRING))}")

# Autokey Cipher
class AutokeyCipher:
    def encode(text: str) -> str:
        pass
    
    def decode(text: str) -> str:
        pass
#print(f"Encoded: {AutokeyCipher.encode(TEST_STRING)}")
#print(f"Decoded: {AutokeyCipher.decode(AutokeyCipher.encode(TEST_STRING))}")

# Beaufort Cipher
class BeaufortCipher:
    def encode(text: str) -> str:
        pass
    
    def decode(text: str) -> str:
        pass
#print(f"Encoded: {BeaufortCipher.encode(TEST_STRING)}")
#print(f"Decoded: {BeaufortCipher.decode(BeaufortCipher.encode(TEST_STRING))}")

# Porta Cipher
class PortaCipher:
    def encode(text: str) -> str:
        pass
    
    def decode(text: str) -> str:
        pass
#print(f"Encoded: {PortaCipher.encode(TEST_STRING)}")
#print(f"Decoded: {PortaCipher.decode(PortaCipher.encode(TEST_STRING))}")

# Running Key Cipher
class RunningKeyCipher:
    def encode(text: str) -> str:
        pass
    
    def decode(text: str) -> str:
        pass
#print(f"Encoded: {RunningKeyCipher.encode(TEST_STRING)}")
#print(f"Decoded: {RunningKeyCipher.decode(RunningKeyCipher.encode(TEST_STRING))}")

# Vigenère and Gronsfeld Cipher
class VigenereAndGronsfeldCipher:
    def encode(text: str) -> str:
        pass
    
    def decode(text: str) -> str:
        pass
#print(f"Encoded: {VigenereAndGronsfeldCipher.encode(TEST_STRING)}")
#print(f"Decoded: {VigenereAndGronsfeldCipher.decode(VigenereAndGronsfeldCipher.encode(TEST_STRING))}")

# Homophonic Substitution Cipher
class HomophonicSubstitutionCipher:
    def encode(text: str) -> str:
        pass
    
    def decode(text: str) -> str:
        pass
#print(f"Encoded: {HomophonicSubstitutionCipher.encode(TEST_STRING)}")
#print(f"Decoded: {HomophonicSubstitutionCipher.decode(HomophonicSubstitutionCipher.encode(TEST_STRING))}")

# Four-Square Cipher
class FourSquareCipher:
    def encode(text: str) -> str:
        pass
    
    def decode(text: str) -> str:
        pass
#print(f"Encoded: {FourSquareCipher.encode(TEST_STRING)}")
#print(f"Decoded: {FourSquareCipher.decode(FourSquareCipher.encode(TEST_STRING))}")

# Hill Cipher
class HillCipher:
    def encode(text: str) -> str:
        pass
    
    def decode(text: str) -> str:
        pass
#print(f"Encoded: {HillCipher.encode(TEST_STRING)}")
#print(f"Decoded: {HillCipher.decode(HillCipher.encode(TEST_STRING))}")

# Playfair Cipher
class PlayfairCipher:
    def encode(text: str) -> str:
        pass
    
    def decode(text: str) -> str:
        pass
#print(f"Encoded: {PlayfairCipher.encode(TEST_STRING)}")
#print(f"Decoded: {PlayfairCipher.decode(PlayfairCipher.encode(TEST_STRING))}")

# ADFGVX Cipher
class ADFGVXCipher:
    def encode(text: str) -> str:
        pass
    
    def decode(text: str) -> str:
        pass
#print(f"Encoded: {ADFGVXCipher.encode(TEST_STRING)}")
#print(f"Decoded: {ADFGVXCipher.decode(ADFGVXCipher.encode(TEST_STRING))}")

# ADFGX Cipher
class ADFGXCipher:
    def encode(text: str) -> str:
        pass
    
    def decode(text: str) -> str:
        pass
#print(f"Encoded: {ADFGXCipher.encode(TEST_STRING)}")
#print(f"Decoded: {ADFGXCipher.decode(ADFGXCipher.encode(TEST_STRING))}")

# Bifid Cipher
class BifidCipher:
    def encode(text: str) -> str:
        pass
    
    def decode(text: str) -> str:
        pass
#print(f"Encoded: {BifidCipher.encode(TEST_STRING)}")
#print(f"Decoded: {BifidCipher.decode(BifidCipher.encode(TEST_STRING))}")

# Straddle Checkerboard Cipher
class StraddleCheckerboardCipher:
    def encode(text: str) -> str:
        pass
    
    def decode(text: str) -> str:
        pass
#print(f"Encoded: {StraddleCheckerboardCipher.encode(TEST_STRING)}")
#print(f"Decoded: {StraddleCheckerboardCipher.decode(StraddleCheckerboardCipher.encode(TEST_STRING))}")

# Trifid Cipher
class TrifidCipher:
    def encode(text: str) -> str:
        pass
    
    def decode(text: str) -> str:
        pass
#print(f"Encoded: {TrifidCipher.encode(TEST_STRING)}")
#print(f"Decoded: {TrifidCipher.decode(TrifidCipher.encode(TEST_STRING))}")

# Base64 Cipher
class Base64Cipher:
    def encode(text: str) -> str:
        pass
    
    def decode(text: str) -> str:
        pass
#print(f"Encoded: {Base64Cipher.encode(TEST_STRING)}")
#print(f"Decoded: {Base64Cipher.decode(Base64Cipher.encode(TEST_STRING))}")

# Fractionated Morse Cipher
class FractionatedMorseCipher:
    def encode(text: str) -> str:
        pass
    
    def decode(text: str) -> str:
        pass
#print(f"Encoded: {FractionatedMorseCipher.encode(TEST_STRING)}")
#print(f"Decoded: {FractionatedMorseCipher.decode(FractionatedMorseCipher.encode(TEST_STRING))}")



# my interactive vigenere cipher
VigenereTable = List[List[str]] # typedef
class Vigenere:
    #VigenereTable = [[None for _ in range(ALPHABET_LENGTH)] for _ in range(ALPHABET_LENGTH)]
    #VigenereTable = List[List[str]] # typedef
    
    def __init__(self,
        keyword: str = "",
        table_filename: str = ""
    ) -> None:
        self.logger = DebugLogger()
        
        # set up crucial vars
        self.alphabet_keyword = keyword
        self.stripped_keyword = Vigenere.remove_duplicates(keyword)
        self.vigenere_table: VigenereTable = [["" for _ in range(ALPHABET_LENGTH)] for _ in range(ALPHABET_LENGTH)]
        # log data 
        self.logger.log(f"Keyword: {self.alphabet_keyword}, stripped keyword: {self.stripped_keyword}",for_debug=False)
        
        # make the keyed alphabet
        self.gen_keyed_alphabet()
        # make the table
        self.gen_vigenere_table(table_filename)
        
        # class OBJs:
        # vigenere_table, keyed_alphabet, keyword, stripped_keyword
    
    def __str__(self) -> str:
        """Outputs the state of the current class instance."""
        
        self.print_table()
        return f"Keyword: {self.alphabet_keyword}\nStripped keyword: {self.stripped_keyword}"
    
    # --- REMOVE DUPLICATES ---------------------
    # pre-condition: string must be a non-empty string of alphabetical characters
    # post-condition: returns a new string without duplicate letters
    @staticmethod
    def remove_duplicates(string: str) -> str:
        """Removes duplicate characters from a string, preserving the order of first occurrence."""
        
        # check if the character is already in the new string before appending
        result = ''
        result = ''.join(char for char in string if char not in result)
        
        return result

    # --- GEN KEYED ALPHABET --------------------
    # pre-condition: keyword must be initialized to a non-empty string of alphabetical characters
    # post-condition: returns a 26-character string containing all unique letters from the keyword
    #                 parameter followed by the remaining English alphabet characters
    def gen_keyed_alphabet(self) -> None:
        """Generates a keyed alphabet by moving the letters of the keyword to the front of the
        standard alphabet and appending the remaining letters. If a keyword is not provided, the
        standard alphabet is used, meaning the keyed alphabet is the standard ASCII alphabet."""
        
        self.logger.log("Entering gen_keyed_alphabet...")
        
        # append the rest of the letters
        # if the letter is not in the keyword, append
        self.keyed_alphabet = self.stripped_keyword + ''.join(char for char in ascii_lowercase if char not in self.stripped_keyword)
        
        # check length is ok, should not be hit?
        if len(self.keyed_alphabet) != ALPHABET_LENGTH:
            self.logger.log("Warning: keyed alphabet is not 26 characters long!",output=stderr)

        self.logger.log(f"Generated keyed alphabet: {self.keyed_alphabet}",for_debug=False)
        self.logger.log("Exiting gen_keyed_alphabet.")
    
    # --- GEN VIGENERE TABLE --------------------
    # pre-condition: if provided, keyed_alphabet must be initialized to a non-empty string of
    #                alphabetical characters, VigenereTable type must be defined. filename string must be initialized to a non-empty string of alphabetical
    #                characters, VigenereTable type must be defined
    # post-condition: returns a newly constructed vigenere table of alphabetical characters based
    #                 on a keyed alphabet. returns a newly constructed vigenere table of alphabetical characters after
    #                 reading from a file
    def gen_vigenere_table(self,
        filename: Optional[str] = None
    ) -> None:
        """Generates a Vigenere table from a keyed alphabet or from a file."""
        
        self.logger.log("Entering gen_vigenere_table...")
        
        # if the filename was provided, use it
        if filename is not None:
            self.logger.log(f"Attempting to read table from file '{filename}'...")
            
            try:
                with open(filename, 'r') as file: # while the file is open
                    for i in range(ALPHABET_LENGTH): # read input
                        line = file.readline().strip() # get each line, store in line var
                        
                        if not line:
                            self.logger.log(f"Error: unable to read line {i+1} from file '{filename}'",output=stderr)
                            return None
                        # for each char in the line
                        for j, char in enumerate(line[:ALPHABET_LENGTH]):
                            # take char from the line and assign to spot in table
                            self.logger.log(f"Current value is {self.vigenere_table[i][j]} at [{i}][{j}]",for_debug=False)
                            self.vigenere_table[i][j] = char
                        # this approach is used instead of the line below so any anomalous chars
                        # and their location can be seen in log
                            
                        #self.vigenere_table[i] = list(line[:ALPHABET_LENGTH])
            # report error if file could not be opened
            except (IOError, FileNotFoundError) as e:
                self.logger.log(f"Error: file '{filename}' unable to be opened - {e}",output=stderr)
                return None
            
        else: # no filename provided
            self.logger.log("Generating a new table...")
            #if self.keyed_alphabet is None: self.keyed_alphabet = ascii_lowercase
            
            # fill in the table
            for i in range(ALPHABET_LENGTH):
                for j in range(ALPHABET_LENGTH):
                    self.vigenere_table[i][j] = self.keyed_alphabet[(i+j) % ALPHABET_LENGTH]
            
        # verify it was generated correctly
        if not self.verify_vigenere_table():
            self.logger.log("Warning: generated vigenere table is invalid!",output=stderr)
        else:
            self.logger.log("Dumping valid table to file.")
            self.dump_vigenere_table()
        
        self.logger.log("Exiting gen_vigenere_table.")

    # --- DUMP VIGENERE TABLE -------------------
    # pre-condition: VigenereTable type must be defined, vigenere_table must be filled with
    #                alphabetical characters (26x26 char matrix), filename string must be initialized
    #                to a non-empty string of alphabetical characters
    # post-condition: nothing is returned, the newly constructed file is created and filled with the
    #                 contents of vigenere_table
    def dump_vigenere_table(self, filename: str = "table") -> None:
        """Dumps the Vigenere table to a file."""
        
        self.logger.log("Entering dump_vigenere_table...")
        
        self.logger.log(f"Writing to filename '{filename}'...")
        try:
            # while the file is open
            with open(filename, 'w') as file:
                # write contents of cipher to file
                for row in self.vigenere_table: file.write(' '.join(row) + "\n")
        # report error if file could not be opened
        except IOError:
            self.logger.log(f"Error: file '{filename}' unable to be opened",output=stderr)
        
        self.logger.log("Exiting dump_vigenere_table.")

    # --- VERIFY VIGENERE TABLE -----------------
    # pre-condition: VigenereTable type must be defined, vigenere_table must be filled with
    #                alphabetical characters (26x26 char matrix)
    # post-condition: true is returned if vigenere_table contains alphabetical characters, otherwise
    #                 false
    def verify_vigenere_table(self) -> bool:
        """Checks that all characters in the Vigenere table are alphabetic."""
        
        self.logger.log("Entering verify_vigenere_table...")
        
        is_valid: bool = True
        
        # check that contents of table are a-z or A-Z
        for i in range(ALPHABET_LENGTH):
            for j in range(ALPHABET_LENGTH):
                if not self.vigenere_table[i][j].isalpha():
                    self.logger.log(f"Invalid character found: {self.vigenere_table[i][j]} at [{i}][{j}]",for_debug=False)
                    is_valid = False
        
        self.logger.log("Exiting verify_vigenere_table.")
        if is_valid: return True
        else: return False

    # --- PRINT VIGENERE TABLE ------------------
    # pre-condition: VigenereTable type must be defined, vigenere_table must be filled with
    #                alphabetical characters (26x26 char matrix)
    # post-condition: if the vigenere table is valid, its contents (in uppercase) are output,
    #                 otherwise only a warning is output
    def print_table(self):
        """Prints the Vigenere table."""
        
        self.logger.log("Entering print_table...")
        
        # first check that the table is valid
        if self.verify_vigenere_table():
            print("\nGenerated table:")
            for row in self.vigenere_table: print(' '.join(char.upper() for char in row))  # output uppercase letters
            
            print("Table is valid")
        else:
            print("Warning: not printing due to table being invalid")
        
        self.logger.log("Exiting print_table.")

    # --- ENCODE --------------------------------
    # pre-condition: VigenereTable type must be defined, vigenere_table must be filled with
    #                alphabetical characters (26x26 char matrix), plaintext and keyword must be
    #                initialized to non-empty strings of alphabetical characters
    # post-condition: the encoded string is returned, including whitespaces, based on the currently
    #                 loaded vigenere table
    def encode(self,
        plaintext: str,
        plaintext_keyword: str # KEYSTREAM
    ) -> str:
        """Encodes plaintext using the Vigenere cipher with the given keyword."""
        
        self.logger.log("Entering encode...")
            
        ciphertext: str = ""
        
        # remove whitespaces, noting any indices
        # update plaintext, store whitespace indices in a list
        self.logger.log(f"Removing whitespaces from plaintext '{plaintext}'...")
        plaintext, whitespace_indices = Vigenere.remove_whitespaces(plaintext)
        
        # ensure keyword length matches plaintext length
        self.logger.log("Adjusting keystream to match plaintext length...")
        keystream = Vigenere.adjust_keystream(plaintext_keyword, len(plaintext))
        
        # for each char in plaintext and keystream
        for p_char, k_char in zip(plaintext, keystream):
            # row index for plaintext char in alphabet
            y_index: int = self.keyed_alphabet.index(p_char)
            # column index keystream char in alphabet
            x_index: int = self.keyed_alphabet.index(k_char)
            # update the ciphertext
            ciphertext += self.vigenere_table[y_index][x_index]
        
        # re-add whitespaces
        self.logger.log("Reinserting whitespaces...")
        ciphertext = Vigenere.reinsert_whitespaces(ciphertext, whitespace_indices)
        
        self.logger.log("Exiting encode.")
        return ciphertext

    # --- DECODE --------------------------------
    # pre-condition: VigenereTable type must be defined, vigenere_table must be filled with
    #                alphabetical characters (26x26 char matrix), ciphertext and keyword must be
    #                initialized to non-empty strings of alphabetical characters
    # post-condition: the decoded string is returned, including whitespaces, based on the currently
    #                 loaded vigenere table
    def decode(self,
        ciphertext: str,
        ciphertext_keyword: str # KEYSTREAM
    ) -> str:
        """Decodes ciphertext using the Vigenere cipher with the given keyword."""
        
        self.logger.log("Entering decode...")
        
        plaintext: str = ""
        
        # remove whitespaces, noting any indices
        # update ciphertext, store whitespace indices in a list
        self.logger.log(f"Removing whitespaces from ciphertext '{ciphertext}'...")
        ciphertext, whitespace_indices = self.remove_whitespaces(ciphertext)
        
        # ensure keyword length matches plaintext length
        self.logger.log("Adjusting keystream to match ciphertext length...")
        keystream = self.adjust_keystream(ciphertext_keyword, len(ciphertext))
        
        # for each char in ciphertext and keystream
        for c_char, k_char in zip(ciphertext, keystream):
            # find pos of keystream char in alphabet
            x_index: int = ascii_lowercase.index(k_char)
            # column index for ciphertext char in vigenere table
            y_index: int = self.vigenere_table[x_index].index(c_char)
            # update the plaintext
            plaintext += ascii_lowercase[y_index]
            #plaintext += ALPHABET[vigenere_table[ALPHABET.index(k_char)].index(c_char)]
        
        # re-add whitespaces
        self.logger.log("Reinserting whitespaces...")
        plaintext = self.reinsert_whitespaces(plaintext, whitespace_indices)
            
        self.logger.log("Exiting decode.")
        return plaintext

    # --- EXTEND KEYSTREAM ----------------------
    # pre-condition: keystream is a non-empty string, target_length is a positive integer greater than
    #                or equal to 0
    # post-condition: returns a string with length equal to target_length. If target_length is greater
    #                 than the length of keystream, the keystream is repeated until the target length
    #                 is reached. If target_length is less than or equal to the length of keystream,
    #                 the keystream is truncated to target_length
    @staticmethod
    def adjust_keystream(
        keyword: str,
        length: int
    ) -> str:
        """Extends or truncates the keyword to match the desired length."""
        
        # if the keyword is shorter than the required length
        if len(keyword) < length:
            # repeat the keyword to meet length
            return (keyword * (length // len(keyword) + 1))[:length]

        # truncate the word to the required length
        return keyword[:length]

    # --- REINSERT WHITESPACES ------------------
    # pre-condition: string is a string without any whitespace characters, whitespace_indices is a list
    #                of integers representing positions in the text where whitespaces were originally
    #                located, whitespace_indices contains valid indices that do not exceed the length
    #                of the final text
    # post-condition: returns a new string with whitespace characters inserted at the specified
    #                 indices. The length of the returned string is equal to
    #                 len(string) + len(whitespace_indices)
    @staticmethod
    def reinsert_whitespaces(
        string: str,
        whitespace_indices: List[int]
    ) -> str:
        """Reinserts whitespaces into the string at the original indices."""
        
        # convert to list for whitespace insertion
        string_list = list(string)
        # add a space at the marked indices
        for index in whitespace_indices: string_list.insert(index, ' ')

        # join back together as one string
        return ''.join(string_list)

    # --- REMOVE WHITESPACES --------------------
    # pre-condition: str must be initialized to a non-empty string of alphabetical characters
    # post-condition: a tuple is returned containing the new string (without whitespaces) and a list
    #                 of the indices of any occuring whitespaces
    @staticmethod
    def remove_whitespaces(string: str) -> Tuple[str, List[int]]:
        """Removes whitespaces from a string, returning the modified string and the indices of removed
        whitespaces."""
        
        # copy var
        string_without_spaces: str = ""
        # list to store indices of all whitespaces
        whitespace_indices: List[int] = []
        
        # for each char in the string
        for i, char in enumerate(string):
            if char.isspace():
                # it is a whitespace, add its index to the list
                whitespace_indices.append(i)
            else:
                # if it is not a space, append to string copy
                string_without_spaces += char
        
        # return a pair containing the new string (without whitespaces) and the list of whitespace
        # indices
        return string_without_spaces, whitespace_indices

# Classic binary encryption
class Binary:
    def binaryEncrypt(binary_str: str) -> str:
        # split binary string into 8 bit chunks (1 byte each)
        binary_values = [binary_str[i:i+8] for i in range(0, len(binary_str), 8)]
        # binary_str[i:i+8]: for each i, split the string into chunks of 8 chars
        
        # convert each byte (binary string) to its corresponding ASCII character
        text = ''.join([chr(int(byte, 2)) for byte in binary_values])
        # int(byte, 2): converts each byte in binary_values to an integer, ',2' notes that it is in
        #               Base-2 (binary)
        # chr(): take the number and convert to it's ASCII character
        
        return text

    def binaryDecrypt(text: str) -> str:
        # convert each character in the text to its binary representation
        binary_str = ''.join([format(ord(char), '08b') for char in text])
        # ord(char): convert each char into its ASCII value
        # format(..., '08b'): converts the ASCII value into an 8-bit binary string
        
        return binary_str

class Hexidecimal:
    pass



#def main():
#    vigenere_table = Vigenere(keyword="kryptos")
#    print(vigenere_table)
#    
#    print(vigenere_table.encode("secret message", "hidden"))
#    
#    print(vigenere_table.decode(vigenere_table.encode("secret message", "hidden"),"hidden"))
#
#
#if __name__ == "__main__":
#    main()


# test binary encryption 
#string = "Hello world"
#binary_string = Encryption.binaryEncrypt(string)
#print(binary_string)
#print(Encryption.binaryDecrypt(binary_string))

# test binary decryption
#test = "Hello! :) I h4v3 2 m4ny pr0j3cts 2 d0 :_("
#test_binary = Encryption.binaryEncrypt(test)
#print(test_binary)
#print(Encryption.binaryDecrypt(test_binary))


