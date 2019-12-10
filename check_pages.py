from random import randint
import argparse
a = [(1,273),(1,299),(1,303)]
with open('2006-2008.txt','a') as f:
    for i,(start,end) in enumerate(a):
        a = [randint(start, end) for a in range(15)]
        a.sort()
        f.write('200'+str(i+6)+'\n')
        f.write(str(a)+'\n')

# if __name__== "__main__" :

#     parser = argparse.ArgumentParser()
#     parser.add_argument('-s','--start')
#     parser.add_argument('-e','--end')
#     parser.add_argument('-t','--total',default = 15, type = int)

#     args = parser.parse_args()
#     start = int(args.start)
#     end = int(args.end)
#     total = int(args.total)

    