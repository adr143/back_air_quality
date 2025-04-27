import serial
import time

# Open serial port
ser = serial.Serial('/dev/serial0', 9600, timeout=5)
time.sleep(1)  # Give module time to respond

# Function to send AT command
def sendAT(command, delay=1):
    ser.write((command + '\r\n').encode())
    time.sleep(delay)
    reply = ser.read_all().decode(errors='ignore')
    print(f"Command: {command}\nReply: {reply}")
    return reply

# Initialize SIM800L
sendAT('AT')             # Test communication
sendAT('AT+CMGF=1')       # Set SMS mode to text

# Send SMS
phone_number = "+639454201486"  # <-- Replace with real phone number
message = "Hello from Raspberry Pi!"

sendAT(f'AT+CMGS="{phone_number}"')
time.sleep(1)  # Wait for '>' prompt
ser.write((message + '\x1A').encode())  # Send message + Ctrl+Z
time.sleep(3)  # Give time to send

# Close the serial
ser.close()
