name= input("what is your name? ")
print ("hello,", name+".",end = " . ") 

print ("hello, \"friend\".",end = " ")
print("hello, world!")


print( "-----------------------------------------------------------")

#strip() remove the white space on head and tail.
name = name.strip()
print (f"hello, {name}")

#title(), cappitalize the string 
name.title()

print (f"hello, {name}")


print( "------------------------------------------------------------------")
date= input( "what is the date today? ").strip().title()
print (date)

print ( "----------------------------------------------------------------")
#split a string 
month,day = date.split(" ")
print (month)