#!/usr/bin/python

import sys
import re
from HTMLParser import HTMLParser

class StromDocParser(HTMLParser):
    html_lines = ""
    __title_start = 0
    __title_attrs = 0
    __h1_start = 0
    __h1_attrs = 0
    __h2_start = 0
    __h2_attrs = 0

    def handle_starttag(self, tag, attrs):
        if (tag == "title"):
            self.__title_start = self.getpos()
            self.__title_attrs = attrs
        elif (tag == "h1"):
            self.__h1_start = self.getpos()
            self.__h1_attrs = attrs
        elif (tag == "h2"):
            self.__h2_start = self.getpos()
            self.__h2_attrs = attrs

    def handle_endtag(self, tag):
        result = ""
        endpos = self.getpos()
        if (tag == "title"):
            startpos = self.__title_start
            attrs = self.__title_attrs
        elif (tag == "h1"):
            startpos = self.__h1_start
            attrs = self.__h1_attrs
        elif (tag == "h2"):
            startpos = self.__h2_start
            attrs = self.__h2_attrs
        else:
            return

        for i in range(startpos[0], endpos[0] + 1, 1):
            line = self.html_lines[i-1]
            if (i == startpos[0] and i == endpos[0]):
                line = line[startpos[1]:endpos[1]]
                line = line[line.find('>') + 1:]
            elif (i == startpos[0]):
                line = line[startpos[1]:]
                line = line[line.find('>') + 1:]
            elif (i == endpos[0]):
                line = line[:endpos[1]]
            result += line
        print tag + ": [" + result + "]"
        for x in attrs:
            print "attrs: " + x[0] + " = " + x[1]

parser = StromDocParser()

del sys.argv[0]
for filename in sys.argv:
    print filename
    fd = open(filename, 'r')
    contents = fd.read()
    parser.html_lines = contents.split('\n')
    parser.feed(contents)
    parser.close()
    parser.reset()
    fd.close()
