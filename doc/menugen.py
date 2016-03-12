#!/usr/bin/python

import sys
import re
import os
import getopt
from HTMLParser import HTMLParser

#
# HTML Parser Enhancement
#
class StromDocParser(HTMLParser):
    html_lines = ""
    menu_items = ""
    __h1_start = 0
    __h1_attrs = 0
    __h2_start = 0
    __h2_attrs = 0
    __h3_start = 0
    __h3_attrs = 0

    def handle_starttag(self, tag, attrs):
        if (tag == "h1"):
            self.__h1_start = self.getpos()
            self.__h1_attrs = attrs
        elif (tag == "h2"):
            self.__h2_start = self.getpos()
            self.__h2_attrs = attrs
        elif (tag == "h3"):
            self.__h3_start = self.getpos()
            self.__h3_attrs = attrs

    def handle_endtag(self, tag):
        result = ""
        endpos = self.getpos()
        if (tag == "h1"):
            startpos = self.__h1_start
            attrs = self.__h1_attrs
        elif (tag == "h2"):
            startpos = self.__h2_start
            attrs = self.__h2_attrs
        elif (tag == "h3"):
            startpos = self.__h3_start
            attrs = self.__h3_attrs
        else:
            return

        for i in range(startpos[0], endpos[0] + 1, 1):
            line = self.html_lines[i-1]
            if (i == startpos[0] and i == endpos[0]):
                line = line[startpos[1]:endpos[1]]
            elif (i == startpos[0]):
                line = line[startpos[1]:]
            elif (i == endpos[0]):
                line = line[:endpos[1]]
            if (len(result) > 0):
                result += " "
            result += line
        result = result[result.find('>') + 1:]

        menuitem = "<li class=\"menuitem_" + tag + "\">"
        menuitem += "<a href=\"./" + self.html_filename
        for x in attrs:
            if (x[0] == "id"):
                menuitem += "#" + x[1]
                break
        menuitem += "\">" + result + "</a></li>\n"
        self.menu_items += menuitem

optlist, args = getopt.getopt(sys.argv[1:], 't:v:m:')

temp_filename = ""
version_number = ""
manual_filename = ""

for x in optlist:
    if (x[0] == '-t'):
        temp_filename = x[1]
    elif (x[0] == '-v'):
        version_number = x[1]
    elif (x[0] == '-m'):
        manual_filename = x[1]
    else:
        print "unexpected option: " + x[0] + " = " + x[1]
        exit(1)

if (temp_filename == "" or version_number == "" or manual_filename == ""):
    print "usage: " + os.path.basename(sys.argv[0]) + "<options> <source files>..."
    print "    -t <template file>"
    print "    -v <version number>"
    print "    -m <manual file>"
    exit(1)

#
# Construction of the table of contents
#
parser = StromDocParser()
for filename in args:
    if (not re.search("\.src\.html$", filename)):
        print "filename postfix is not expected :" + filename
        exit(1)
    fd = open(filename, 'r')
    contents = fd.read()
    contents = re.sub('\r', '', contents)
    parser.html_filename = re.sub("\.src\.html$", ".html", filename)
    parser.html_lines = contents.split('\n')
    parser.feed(contents)
    parser.close()
    parser.reset()
    fd.close()
menu_items = parser.menu_items

#
# Apply source HTML to the template
#
fd = open(temp_filename, "r")
template = fd.read()
fd.close()

template = template.replace("%%%PGSTROM_MANUAL_VERSION%%%", version_number)
template = template.replace("%%%PGSTROM_MANUAL_MENU%%%", menu_items)
fd = open(manual_filename, "r")
template = template.replace("%%%PGSTROM_MANUAL_BODY%%%", fd.read())
fd.close()

print template
