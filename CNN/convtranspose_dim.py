from math import floor
h=input("h") #height
k=input("k") #kernel 
s=input("s") #stride 
p=input("p") #padding 



print(floor(s*(h-1)+k-2*p))