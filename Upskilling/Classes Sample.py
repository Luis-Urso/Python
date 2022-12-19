# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 19:05:22 2022

@author: WB02554
"""

## How to make a new class (object)

class calcs:
    def __init__(self,a,b=None):
        self.a=a
        if b==None:b=0 
        self.b=b
        self.c=a+b
        
    def summary(self):
        return(self.a+self.b)
    
    def subtract(self):
        return(self.a-self.b)
    
    def circle(self):
        return(self.a*self.a*3.1415)
    
    
    
# calc = calcs(10,20)

# print (calc.summary())

# print (calc.subtract())

# print (calc.a)

# print (calc.c)


calc1=calcs(10,10)
calc2=calcs(20,20)
calc3=calcs(10)

# Call the methods (inner object functions)

print(calc1.summary())
print(calc2.summary())
print(calc3.circle())

# Call the attributes (inned object variables)

print(calc1.a)
print(calc2.a)





