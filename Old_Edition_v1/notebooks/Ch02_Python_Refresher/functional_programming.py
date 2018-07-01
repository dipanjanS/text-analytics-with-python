# -*- coding: utf-8 -*-
"""
Created on Wed Oct 05 01:38:23 2016

@author: DIP
"""

# function with single argument
def square(number):
    return number*number

square(5)

# built-in function from the numpy library
import numpy as np
np.square(5)

# a more complex function with variable number of arguments
def squares(*args):
    squared_args = []
    for item in args: 
        squared_args.append(item*item)
    return squared_args

squares(1,2,3,4,5)


# assign specific keyword based arguments dynamically
def person_details(**kwargs):
    for key, value in kwargs.items():
        print key, '->', value
      
person_details(name='James Bond', alias='007', job='Secret Service Agent')


# using recursion to square numbers
def recursive_squares(numbers):
    if not numbers:
        return []
    else:
        return [numbers[0]*numbers[0]] + recursive_squares(numbers[1:])

recursive_squares([1, 2, 3, 4, 5])


# simple lambda function to square a number
lambda_square = lambda n: n*n
lambda_square(5)

# map function to square numbers using lambda
map(lambda_square, [1, 2, 3, 4, 5])

# lambda function to find even numbers used for filtering
lambda_evens = lambda n: n%2 == 0
filter(lambda_evens, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# lambda function to add numbers used for adding numbers in reduce function
lambda_sum = lambda x, y: x + y
reduce(lambda_sum, [1, 2, 3, 4, 5])

# lambda function to make a sentence from word tokens with reduce function
lambda_sentence_maker = lambda word1, word2: ' '.join([word1, word2])
reduce(lambda_sentence_maker, ['I', 'am', 'making', 'a', 'sentence', 'from', 'words!'])


# iterators

# typical for loop
numbers = range(6)
for number in numbers:
    print number

# illustrating how iterators work behind the scenes
iterator_obj = iter(numbers)
while True:
    try:
        print iterator_obj.next()
    except StopIteration:
        print 'Reached end of sequence'
        break

# calling next now would throw the StopIteration exception as expected
iterator_obj.next()


# comprehensions
numbers = range(6)
numbers

# simple list comprehension to compute squares
[num*num for num in numbers]

# list comprehension to check if number is divisible by 2
[num%2 for num in numbers]

# set comprehension returns distinct values of the above operation
set(num%2 for num in numbers)

# dictionary comprehension where key:value is number: square(number)
{num: num*num for num in numbers}

# a more complex comprehension showcasing above operations in a single comprehension
[{'number': num, 
  'square': num*num, 
  'type': 'even' if num%2 == 0 else 'odd'} for num in numbers]

# nested list comprehension - flattening a list of lists
list_of_lists = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
list_of_lists
[item for each_list in list_of_lists for item in each_list]



# generators
numbers = [1, 2, 3, 4, 5]

def generate_squares(numbers):
    for number in numbers:
        yield number*number

gen_obj = generate_squares(numbers)
gen_obj
for item in gen_obj:
    print item

csv_string = 'The,fox,jumps,over,the,dog'
# making a sentence using list comprehension
list_cmp_obj = [item for item in csv_string.split(',')]
list_cmp_obj
' '.join(list_cmp_obj)

# making a sentence using generator expression
gen_obj = (item for item in csv_string.split(','))
gen_obj
' '.join(gen_obj)


     




