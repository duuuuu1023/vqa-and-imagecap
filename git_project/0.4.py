str="Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."
strlst=str.split()
one=[1,5,6,7,8,9,15,16,19]
x={}
for i,word in enumerate(strlst):
    if i+1 in one:
        x[word[:1]]=i+1
    else:
        x[word[:2]]=i+1
print(x)