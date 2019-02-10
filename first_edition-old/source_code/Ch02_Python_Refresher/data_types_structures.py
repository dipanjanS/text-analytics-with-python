# -*- coding: utf-8 -*-
"""
Created on Wed Oct 05 00:28:31 2016

@author: DIP
"""

# zen of python
import this

new_string = "This is a String"  # storing a string

id(new_string)  # shows the object identifier (address)

type(new_string)  # shows the object type

new_string  # shows the object value


# representing integers and operations on them
num = 123

type(num)

num + 1000  # addition

num * 2  # multiplication

num /  2  # integer division


# decimal
1 + 1

# binary
bin(2)

0b1 + 0b1

bin(0b1 + 0b1)

# octal
oct(8)

oct(07 + 01)

0o10


# hexadecimal
hex(16)

0x10

hex(0x16 + 0x5)


# floating points
1.5 + 2.6

1e2 + 1.5e3 + 0.5

2.5e4

2.5e-2


# complex
cnum = 5 + 7j

type(cnum)

cnum.real

cnum.imag

cnum + (1 - 0.5j)


# strings
s1 = 'this is a string'
s2 = 'this is "another" string'
s3 = 'this is the \'third\' string'
s4 = """this is a
multiline
string"""

print s1, s2, s3, s4

print s3 + '\n' + s4


' '.join([s1, s2])

s1[::-1]  # reverses the string


# lists
l1 = ['eggs', 'flour', 'butter']
l2 = list([1, 'drink', 10, 'sandwiches', 0.45e-2])
l3 = [1, 2, 3, ['a', 'b', 'c'], ['Hello', 'Python']]

print l1, l2, l3

# indexing lists
l1
l1[0]
l1[1]
l1[0] +' '+ l1[1]

# slicing lists
l2[1:3]

numbers = range(10)
numbers
numbers[2:5]
numbers[:]
numbers[::2]

# concatenating and mutating lists
numbers * 2
numbers + l2

# handling nested lists
l3
l3[3]
l3[4]
l3.append(' '.join(l3[4]))  # append operation
l3
l3.pop(3)  # pop operation
l3



# sets
l1 = [1,1,2,3,5,5,7,9,1]

set(l1)  # makes the list as a set
s1 = set(l1)

# membership testing
1 in s1  
100 in s1

# initialize a second set
s2 = {5, 7, 11}

# testing various set operations
s1 - s2  # set difference
s1 | s2  # set union
s1 & s2  # set intersection 
s1 ^ s2  # elements which do not appear in both sets


# dictionaries
d1 = {'eggs': 2, 'milk': 3, 'spam': 10, 'ham': 15}
d1

# retrieving items based on key
d1.get('eggs')
d1['eggs']

# get is better than direct indexing since it does not throw errors
d1.get('orange') 
d1['orange']

# setting items with a specific key
d1['orange'] = 25
d1

# viewing keys and values
d1.keys()
d1.values()

# create a new dictionary using dict function
d2 = dict({'orange': 5, 'melon': 17, 'milk': 10})
d2

# update dictionary d1 based on new key-values in d2
d1.update(d2)
d1

# complex and nested dictionary
d3 = {'k1': 5, 'k2': [1,2,3,4,5], 'k3': {'a': 1, 'b': 2, 'c': [1,2,3]}}
d3
d3.get('k3')
d3.get('k3').get('c')


# tuples

# creating a tuple with a single element 
single_tuple = (1,)
single_tuple

# original address of the tuple
id(single_tuple)

# modifying contents of the tuple but its location changes (new tuple is created)
single_tuple = single_tuple + (2, 3, 4, 5)
single_tuple
id(single_tuple) # different address indicating new tuple with same name

# tuples are immutable hence assignment is not supported like lists
single_tuple[3] = 100

# accessing and unpacking tuples
tup = (['this', 'is', 'list', '1'], ['this', 'is', 'list', '2'])
tup[0]
l1, l2 = tup
print l1, l2



# files
f = open('text_file.txt', 'w')   # open in write mode
f.write("This is some text\n")  # write some text
f.write("Hello world!")
f.close()  # closes the file

# lists files in current directory
import os
os.listdir(os.getcwd())

f = open('text_file.txt', 'r')  # opens file in read mode
data = f.readlines()  # reads in all lines from file
print data  # prints the text data



