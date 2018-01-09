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
    toc_index = []
    html_lines = ""
    menu_items = ""
    __section_id = ""

    __htag_start = []
    __htag_attrs = []

    def handle_starttag(self, tag, attrs):
        if (tag == "section" or tag == "article"):
            for x in attrs:
                if (x[0] == "id"):
                    self.__section_id = x[1]
                    break
        elif (len(self.__section_id) > 0 and \
              (tag == "h1" or tag == "h2" or tag == "h3" or tag == "h4")):
            self.__htag_kind = tag
            self.__htag_start = self.getpos()

    def handle_endtag(self, tag):
        if (len(self.__section_id) > 0 and \
            (tag == "h1" or tag == "h2" or tag == "h3" or tag == "h4")):
            startpos = self.__htag_start
            endpos = self.getpos()
        else:
            return

        result = ""
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

        link = "<a href=\"./" + os.path.basename(self.html_filename)
        if (self.__htag_kind != "h1"):
            link += "#" + self.__section_id
        link += "\">" + result + "</a>"
        self.toc_index.append([self.__section_id, link])
        if (self.__htag_kind == "h1" or self.__htag_kind == "h2"):
            self.menu_items += "<li class=\"menuitem_"+tag+"\">"+link+"</li>\n"
        self.__section_id = ""	# reset

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
toc_index = parser.toc_index

#
# Apply source HTML to the template
#
fd = open(temp_filename, "r")
template = fd.read()
fd.close()

fd = open(manual_filename, "r")
manual = fd.read()
fd.close()

#
# Replace PGSTROM_MANUAL_XLINK by <a href=...> tag
#
for x in toc_index:
    pattern = "%%%PGSTROM_MANUAL_XLINK:" + x[0] + "%%%"
    manual = manual.replace(pattern, x[1])

#
# Insert manual into the template
#
template = template.replace("%%%PGSTROM_MANUAL_VERSION%%%", version_number)
template = template.replace("%%%PGSTROM_MANUAL_MENU%%%", menu_items)
template = template.replace("%%%PGSTROM_MANUAL_BODY%%%", manual)

print template
