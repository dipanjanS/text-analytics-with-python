# -*- coding: utf-8 -*-
"""
Created on Wed Oct 05 01:58:37 2016

@author: DIP
"""


# String types

# simple string
simple_string = 'hello' + " I'm a simple string"
print simple_string

# multi-line string, note the \n (newline) escape character automatically created
multi_line_string = """Hello I'm
a multi-line
string!"""
multi_line_string
print multi_line_string

# Normal string with escape sequences leading to a wrong file path!
escaped_string = "C:\the_folder\new_dir\file.txt"
print escaped_string  # will cause errors if we try to open a file here

# raw string keeping the backslashes in its normal form
raw_string = r'C:\the_folder\new_dir\file.txt'
print raw_string

# unicode string literals
string_with_unicode = u'H\u00e8llo!'
print string_with_unicode



# String operations

# Different ways of String concatenation
'Hello' + ' and welcome ' + 'to Python!'
'Hello' ' and welcome ' 'to Python!'

# concatenation of variables and literals
s1 = 'Python!'
'Hello ' + s1

# we cannot concatenate a variable and a literal using this method
'Hello ' s1

# some more ways of concatenating strings
s2 = '--Python--'
s2 * 5
s1 + s2
(s1 + s2)*3
'Python!--Python--Python!--Python--Python!--Python--'

# concatenating several strings together in parentheses
s3 = ('This '
      'is another way '
      'to concatenate '
      'several strings!')
s3

# checking for substrings in a string
'way' in s3
'python' in s3

# computing total length of the string
len(s3)



# String indexing and slicing

# creating a string
s = 'PYTHON'

# depicting string indexes
for index, character in enumerate(s):
    print 'Character', character+':', 'has index:', index

# string indexing
s[0], s[1], s[2], s[3], s[4], s[5]
s[-1], s[-2], s[-3], s[-4], s[-5], s[-6]

# string slicing
s[:] 
s[1:4]
s[:3]
s[3:]
s[-3:]
s[:3] + s[3:]
s[:3] + s[-3:]

# string slicing with offsets
s[::1]  # no offset
s[::2]  # print every 2nd character in string

# strings are immutable hence assignment throws error
s[0] = 'X'

# creates a new string
'X' + s[1:]



# String methods

# case conversions
s = 'python is great'
s.capitalize()
s.upper()

# string replace
s.replace('python', 'analytics')

# string splitting and joining
s = 'I,am,a,comma,separated,string'
s.split(',')
' '.join(s.split(','))

# stripping whitespace characters
s = '   I am surrounded by spaces    '
s
s.strip()
'I am surrounded by spaces'

# coverting to title case
s = 'this is in lower case'
s.title()



# String formatting

# simple string formatting expressions
'Hello %s' %('Python!')
'Hello %s' %('World!')

# formatting expressions with different data types
'We have %d %s containing %.2f gallons of %s' %(2, 'bottles', 2.5, 'milk')
'We have %d %s containing %.2f gallons of %s' %(5, 'jugs', 10.867, 'juice')

# formatting using the format method
'Hello {} {}, it is a great {} to meet you'.format('Mr.', 'Jones', 'pleasure')
'Hello {} {}, it is a great {} to meet you'.format('Sir', 'Arthur', 'honor')

# alternative ways of using format
'I have a {food_item} and a {drink_item} with me'.format(drink_item='soda', food_item='sandwich')
'The {animal} has the following attributes: {attributes}'.format(animal='dog', attributes=['lazy', 'loyal'])



# Using regular expressions

# importing the re module
import re

# dealing with unicode matching using regexes
s = u'H\u00e8llo'
s
print s

# does not return the special unicode character even if it is alphanumeric
re.findall(r'\w+', s)

# need to explicitely specify the unicode flag to detect it using regex
re.findall(r'\w+', s, re.UNICODE)

# setting up a pattern we want to use as a regex
# also creating two sample strings
pattern = 'python'
s1 = 'Python is an excellent language'
s2 = 'I love the Python language. I also use Python to build applications at work!'

# match only returns a match if regex match is found at the beginning of the string
re.match(pattern, s1)
# pattern is in lower case hence ignore case flag helps
# in matching same pattern with different cases
re.match(pattern, s1, flags=re.IGNORECASE)

# printing matched string and its indices in the original string
m = re.match(pattern, s1, flags=re.IGNORECASE)
print 'Found match {} ranging from index {} - {} in the string "{}"'.format(m.group(0), m.start(), m.end(), s1)

# match does not work when pattern is not there in the beginning of string s2
re.match(pattern, s2, re.IGNORECASE)

# illustrating find and search methods using the re module
re.search(pattern, s2, re.IGNORECASE)
re.findall(pattern, s2, re.IGNORECASE)

match_objs = re.finditer(pattern, s2, re.IGNORECASE)
print "String:", s2
for m in match_objs:
    print 'Found match "{}" ranging from index {} - {}'.format(m.group(0), m.start(), m.end())    

# illustrating pattern substitution using sub and subn methods
re.sub(pattern, 'Java', s2, flags=re.IGNORECASE)
re.subn(pattern, 'Java', s2, flags=re.IGNORECASE)

