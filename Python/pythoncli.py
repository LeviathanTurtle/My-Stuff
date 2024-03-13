"""
// main.c
#include <stdio.h>

int main(int argc, char *argv[]) {
    printf("Arguments count: %d\n", argc);
    for (int i = 0; i < argc; i++) {
        printf("Argument %6d: %s\n", i, argv[i]);
    }
    return 0;
}
"""



# main.py
import sys

if __name__ == "__main__":
    print(f"Arguments count: {len(sys.argv)}")
    for i, arg in enumerate(sys.argv):
        print(f"Argument {i:>6}: {arg}")