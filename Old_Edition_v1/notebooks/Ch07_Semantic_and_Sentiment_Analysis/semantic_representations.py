# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 18:46:04 2016

@author: DIP
"""

import nltk
import pandas as pd
import os

symbol_P = 'P'
symbol_Q = 'Q'
proposition_P = 'He is hungry'
propositon_Q = 'He will eat a sandwich'

p_statuses = [False, False, True, True]
q_statuses = [False, True, False, True]

conjunction = '(P & Q)'
disjunction = '(P | Q)'
implication = '(P -> Q)'
equivalence = '(P <-> Q)'
expressions = [conjunction, disjunction, implication, equivalence]


results = []
for status_p, status_q in zip(p_statuses, q_statuses):
    dom = set([])
    val = nltk.Valuation([(symbol_P, status_p), 
                          (symbol_Q, status_q)])
    assignments = nltk.Assignment(dom)
    model = nltk.Model(dom, val)
    row = [status_p, status_q]
    for expression in expressions:
        result = model.evaluate(expression, assignments)
        row.append(result)
    results.append(row)
    
columns = [symbol_P, symbol_Q, conjunction, 
           disjunction, implication, equivalence]           
result_frame = pd.DataFrame(results, columns=columns)

print 'P:', proposition_P
print 'Q:', propositon_Q
print
print 'Expression Outcomes:-'
print result_frame   



# first order logic

read_expr = nltk.sem.Expression.fromstring

os.environ['PROVER9'] = r'E:/prover9/bin'
prover = nltk.Prover9()

prover = nltk.ResolutionProver()   

# set the rule expression
rule = read_expr('all x. all y. (jumps_over(x, y) -> -jumps_over(y, x))')
# set the event occured
event = read_expr('jumps_over(fox, dog)')
# set the outcome we want to evaluate -- the goal
test_outcome = read_expr('jumps_over(dog, fox)')

# get the result
prover.prove(goal=test_outcome, 
             assumptions=[event, rule],
             verbose=True)

# set the rule expression                          
rule = read_expr('all x. (studies(x, exam) -> pass(x, exam))') 
# set the events and outcomes we want to determine
event1 = read_expr('-studies(John, exam)')  
test_outcome1 = read_expr('pass(John, exam)') 
event2 = read_expr('studies(Pierre, exam)')  
test_outcome2 = read_expr('pass(Pierre, exam)') 

prover.prove(goal=test_outcome1, 
             assumptions=[event1, rule],
             verbose=True)  
             
prover.prove(goal=test_outcome2, 
             assumptions=[event2, rule],
             verbose=True)               
             
             
          
 
# define symbols (entities\functions) and their values
rules = """
    rover => r
    felix => f
    garfield => g
    alex => a
    dog => {r, a}
    cat => {g}
    fox => {f}
    runs => {a, f}
    sleeps => {r, g}
    jumps_over => {(f, g), (a, g), (f, r), (a, r)}
    """
val = nltk.Valuation.fromstring(rules)

print val

dom = {'r', 'f', 'g', 'a'}
m = nltk.Model(dom, val)

print m.evaluate('jumps_over(felix, rover) & dog(rover) & runs(rover)', None)
print m.evaluate('jumps_over(felix, rover) & dog(rover) & -runs(rover)', None)
print m.evaluate('jumps_over(alex, garfield) & dog(alex) & cat(garfield) & sleeps(garfield)', None)


g = nltk.Assignment(dom, [('x', 'r'), ('y', 'f')])   
print m.evaluate('runs(y) & jumps_over(y, x) & sleeps(x)', g)   
print m.evaluate('exists y. (fox(y) & runs(y))', g)     


formula = read_expr('runs(x)')
print m.satisfiers(formula, 'x', g)  

formula = read_expr('runs(x) & fox(x)')
print m.satisfiers(formula, 'x', g)              
             
             