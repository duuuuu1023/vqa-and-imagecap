str="Now I need a drink,aloholic of course,after the heavy lectures involving quantum mechanics"
strlst=str.split(",")
lstNum=[]
for i in iter(strlst):
    strlstS=i.split()
    for j in iter(strlstS):
        lstNum.append(len(j))
print(lstNum)