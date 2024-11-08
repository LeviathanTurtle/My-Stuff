
# 
# 
# 

def binary_to_text(binary_str: str) -> str:
    # split binary string into 8 bit chunks (1 byte each)
    binary_values = [binary_str[i:i+8] for i in range(0, len(binary_str), 8)]
    # binary_str[i:i+8]: for each i, split the string into chunks of 8 chars
    
    # convert each byte (binary string) to its corresponding ASCII character
    text = ''.join([chr(int(byte, 2)) for byte in binary_values])
    # int(byte, 2): converts each byte in binary_values to an integer, ',2' notes that it is in
    #               Base-2 (binary)
    # chr(): take the number and convert to it's ASCII character
    
    return text

def text_to_binary(text: str) -> str:
    # convert each character in the text to its binary representation
    binary_str = ''.join([format(ord(char), '08b') for char in text])
    # ord(char): convert each char into its ASCII value
    # format(..., '08b'): converts the ASCII value into an 8-bit binary string
    
    return binary_str


def main():
    # test binary encryption 
    string = "Hello world"
    binary_string = text_to_binary(string)
    print(binary_string)
    print(binary_to_text(binary_string))
    
    # test binary decryption
    test = "Hello! :) I h4v3 2 m4ny pr0j3cts 2 d0 :_("
    test_binary = text_to_binary(test)
    print(test_binary)
    print(binary_to_text(test_binary))

if __name__ == "__main__":
    main()