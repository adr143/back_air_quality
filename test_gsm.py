import time
from sim800l import SIM800L

sim800l=SIM800L('/dev/serial0')

# Send data and return the first line
print(sim800l.command("AT+CCLK?\n"))  # ...+CCLK...

# Same as before, but reading both lines
sim800l.command("AT+CCLK?\n", lines=0)  # send AT command without reading data
print("First read line:", sim800l.check_incoming())  # ...+CCLK...
print("Second read line:", sim800l.check_incoming())  # ...'OK'...

# Same as before, but more elaborated
sim800l.command("AT+CCLK?\n", lines=0)
expire = time.monotonic() + 2  # seconds
sequence = ""
s = sim800l.check_incoming()
date = None
while time.monotonic() < expire:
    if s[0] == 'GENERIC' and s[1] and s[1].startswith('+CCLK: "'):
        date = s[1].split('"')[1]
        sequence += "D"
    if s == ('OK', None):
        sequence += "O"
    if sequence == "DO":
        print("Date:", date)
        break
    time.sleep(0.1)
    s = sim800l.check_incoming()

if not date:
    print("Error")
