a = int(input("Enter First Number : "))
b = int(input("Enter Second Number : "))
n = int(input("Enter Number : "))

print(a,b,end=" ")

while(n-2):
	c=a+b
	a=b
	b=c
	print(c,end=" ")
	n=n-1
	
