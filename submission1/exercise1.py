i = 0
dict = {"Iris-setosa":'1', "Iris-versicolor": '2', "Iris-virginica": '3'}
fname = 'iris.data'
fout = open ('iris-svm-input.txt', 'w')
with open (fname) as f:
	content = f.readlines()
content = [line[:-1] for line in content]
#print (content)
for x in content:
	i = 0
	words = x.split(",")
	if len(words) < 2:
		break
	while i < 4:
		if words[i] == 0:
			break;
		else:
			fout.write (dict[words[4]]+' '+str(i+1)+':'+words[i]+' ')
		i = i+1
	fout.write('\n')