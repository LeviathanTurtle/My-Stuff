
# 
# William Wadsworth
# 03.08.2025
# 

from package import Package

# pre-condition: 
# post-condition: 
def process_scan(scan_data: str):
    """function def."""
    
    print(f"Package scanned: {scan_data}")
    
    if scan_data.startswith("PKG"): print("Valid package ID detected")
    else: print("Invalid scan. Please try again")



def main():
    print("Scanner active")
    
    # todo: check hardware link BEFORE proceeding
    
    while True:
        try:
            scan_data = input("Waiting for package scan...")
            
            if scan_data.lower() == "exit":
                print("Exiting scanner system")
                break
            
            process_scan(scan_data)
        except KeyboardInterrupt:
            print("\nScanner system shutting down")
            break

if __name__ == '__main__':
    main()