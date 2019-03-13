#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 11:13:54 2019

@author: amith
DOSto UNix
"""

#!/usr/bin/env python
"""\
convert dos linefeeds (crlf) to unix (lf)
usage: dos2unix.py <input> <output>
"""
import sys

if len(sys.argv[1:]) != 2:
  sys.exit(__doc__)

content = ''
outsize = 0
with open(sys.argv[1], 'rb') as infile:
  content = infile.read()
with open(sys.argv[2], 'wb') as output:
  for line in content.splitlines():
    outsize += len(line) + 1
    output.write(line + '\n')

print("Done. Saved %s bytes." % (len(content)-outsize)
