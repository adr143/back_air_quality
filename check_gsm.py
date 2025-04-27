import serial
import time

ser = serial.Serial('/dev/serial0', 115200, timeout=2)  # or try 115200 if no response
time.sleep(2)

ser.write(b'AT\r\n')
time.sleep(1)

response = ser.read_all().decode(errors='ignore')
print(f"SIM800L Response: {response}")

ser.close()

