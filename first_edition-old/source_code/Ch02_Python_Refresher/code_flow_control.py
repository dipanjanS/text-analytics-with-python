# -*- coding: utf-8 -*-
"""
Created on Wed Oct 05 01:32:15 2016

@author: DIP
"""

# if, if-elif, if-elif-else
var = 'spam'
if var == 'spam':
    print 'Spam'

var = 'ham'
if var == 'spam':
    print 'Spam'
elif var == 'ham':
    print 'Ham'


var = 'foo'
if var == 'spam':
    print 'Spam'
elif var == 'ham':
    print 'Ham'
else: 
    print 'Neither Spam or Ham'



# Looping constructs

# illustrating for loops
numbers = range(0,5)
for number in numbers:
    print number

sum = 0
for number in numbers:
    sum += number

print sum


# role of the trailing else and break constructs
for number in numbers:
    print number
else:
    print 'loop exited normally'


for number in numbers:
    if number < 3:
        print number
    else:
        break
else:
    print 'loop exited normally'

# illustrating while loops
number = 5
while number > 0:
    print number
    number -= 1  # important! else loop will keep running

# role of continue construct
number = 10
while number > 0:
    if number % 2 != 0:
        number -=1 # decrement but do not print odd numbers
        continue  # go back to beginning of loop for next iteration
    print number  # print even numbers and decrement count
    number -= 1  

# role of the pass construct
number = 10
while number > 0:
    if number % 2 != 0:
        pass # don't print odds
    else:
        print number
    number -= 1



# exceptions
shopping_list = ['eggs', 'ham', 'bacon']
# trying to access a non-existent item in the list
try:
    print shopping_list[3]
except IndexError as e:
    print 'Exception: '+str(e)+' has occured'
else:
    print 'No exceptions occured'
finally:
    print 'I will always execute no matter what!'
    
# smooth code execution without any errors
try:
    print shopping_list[2]
except IndexError as e:
    print 'Exception: '+str(e)+' has occured'
else:
    print 'No exceptions occured'
finally:
    print 'I will always execute no matter what!'
 


