# 
# 
# 

import qrcode
import qrcode.constants

def generateQRCode(text, filename):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(text)
    qr.make()
    img = qr.make_image(fill_color="#6aff9b",back_color="black")
    img.save(filename)

def main():
    text = input("Enter the URL: ")
    filename = "qrcode.png"
    generateQRCode(text,filename)
    print(f"QR code saved as {filename}")

if __name__ == "__main__":
    main()