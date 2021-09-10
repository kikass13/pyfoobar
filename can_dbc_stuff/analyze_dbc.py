import sys
import cantools
from pprint import pprint

# pip3 install cantools


db = cantools.database.load_file(sys.argv[1])
print("Message attributes look like this:")
print("name, frame_id, is_extended_frame, length, comments")
print("Signal attributes look like this:")
print("name, start, length, byte_order, is_signed, initial, scale, offset, minimum, maximum, unit, is_multiplexer, multiplexer_ids, spn[The J1939 Suspect Parameter Number], comments")

print("")
pprint(db.messages)
print("==================================")
for msg in db.messages:
   print("%s: " % msg._name)
   pprint(msg.signals)
   print("")