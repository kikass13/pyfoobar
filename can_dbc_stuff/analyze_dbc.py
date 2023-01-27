import sys
import math
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

### create a string tree representation of the dbc

### signal to random type repr
# ###
#                 if constexpr (sizeof(T) <= 4) {
#                     t[d] = 'f';
#                 } else if (sizeof(T) == 8) {
#                     t[d] = 'd';
#                 }
#             } else if constexpr (std::is_integral<T>::value) {
#                 if constexpr (std::is_unsigned<T>::value) {
#                     if constexpr (sizeof(T) == 1) {
#                         t[d] = 'B';
#                     } else if (sizeof(T) == 2) {
#                         t[d] = 'H';
#                     } else if (sizeof(T) == 4) {
#                         t[d] = 'I';
#                     } else if (sizeof(T) == 8) {
#                         t[d] = 'Q';
#                     }
#                 } else {
#                     if constexpr (sizeof(T) == 1) {
#                         t[d] = 'b';
#                     } else if (sizeof(T) == 2) {
#                         t[d] = 'h';
#                     } else if (sizeof(T) == 4) {
#                         t[d] = 'i';
#                     } else if (sizeof(T) == 8) {
#                         t[d] = 'q';
#                     }
#                 }
def signalRepr(signal : cantools.database.can.signal.Signal):
   name = signal.name
   length = signal.length
   bytelength = math.ceil(length / 8)
   t = "*"
   if signal.is_float:
      if bytelength <= 4:
         t = "f"
      else:
         t = "d"
   else:
      if bytelength == 1: ###bool
         t = "b"
      elif bytelength == 2:
         t = "h"
      elif bytelength == 4:
         t = "i"
      elif bytelength == 8:
         t = "q"
      ## signed(lowercase) / unsigned (uppercase)
      if not signal.is_signed:
         t = t.upper()
   description = "%s:%s:%s" % (name, bytelength, t)
   return description

tree = ""
for msg in db.messages:
   signals = []
   for s in msg.signals:
      signals.append(signalRepr(s))
   tree += "{%s:r %s}" % (msg._name, " ".join(signals))
print("some parsable nltk string: ")
print(tree)
