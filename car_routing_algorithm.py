# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 23:14:32 2018

@author: mhari
"""
import time
import numpy as np
class Car:
    gain=[0 for i in range(10)]
    mean_gain=0
    utility=[]
    policy=[]
    reward=[]
    def __init__(self,start,end):
        self.start=start
        self.end=end

def printer(lis):
    for i in lis:
        print(i)
def location_extract(x,start,end):
    res=[]
    for i in range(start,end):
        a,b=tuple(x[i])
        res.append(tuple((b,a)))
#    print(res)
    return res
def board_gen(s,obstacles):
    board=[[-1.0 for i in range(s)] for j in range(s)]
    for i in obstacles:
        a,b=i
        board[a][b]=np.float64(-101)
    return board
def parse_input():
    filename="gen.txt"
    fopen=open(filename,"r")
    input=fopen.readlines()
    input_lis=[i.rstrip("\n").split(",") for i in input]
    fopen.close()
     
    for i,ival in enumerate(input_lis):
        for k,kval in enumerate(ival):
            input_lis[i][k]=int(kval)
    s,n,o = input_lis[:3]
    s=s[0]
    n=n[0]
    o=o[0]
    counter=3
    obstacles=location_extract(input_lis,counter,counter+o)
    counter+=o
    start_list=location_extract(input_lis,counter,counter+n)
    counter+=n
    end_list=location_extract(input_lis,counter,counter+n)
    cars=[]
    for i in range(n):
        cars.append(Car(start_list[i],end_list[i]))  
    board=board_gen(s,obstacles)
    return (s,board,cars)


def reward_gen(reward,car):
    a,b=car.end
    reward[a][b]=np.float64(99)
    return reward

def mapping(board_len):
    mapper={}
    for i in range(board_len):
        for j in range(board_len):
            mapper[(i,j,"n")]=(i,j)
            mapper[(i,j,"s")]=(i,j)
            mapper[(i,j,"w")]=(i,j)
            mapper[(i,j,"e")]=(i,j)
            if i>0:               
                mapper[(i,j,"n")]=(i-1,j)
            if j>0:                
                mapper[(i,j,"w")]=(i,j-1)
            if i<board_len-1:               
                mapper[(i,j,"s")]=(i+1,j)
            if j<board_len-1:
                mapper[(i,j,"e")]=(i,j+1)
    return mapper


def utility_gen(board_len,car,gamma,delta):
    reward=car.reward
    util_curr=[x[:] for x in reward]
    util_prev=[x[:] for x in reward]

    a,b=car.end
    while True:

        residue=0

        for i in range(board_len):
            for j in range(board_len):
                if (a,b)==(i,j):
                    continue
                n=util_prev[i][j]
                s=n
                w=n
                e=n

                if i>0:
                    n=util_prev[i-1][j]
                   
                if j>0:
                    w=util_prev[i][j-1]
                    
                if i<board_len-1:
                    s=util_prev[i+1][j]
                    
                if j<board_len-1:
                    e=util_prev[i][j+1]

                left=0.7*w+0.1*(n+s+e)
                right=0.7*e+0.1*(n+s+w)
                up=0.7*n+0.1*(w+s+e)
                down=0.7*s+0.1*(n+w+e) 
                util_curr[i][j]=reward[i][j]+gamma*max(left,right,up,down)
                temp=abs(util_curr[i][j]-util_prev[i][j])
                residue=max(temp,residue)

        
        if residue<delta:
            return util_curr
        util_prev=[x[:] for x in util_curr]
        


def policy_gen(board_len,util):
    policy=[[() for i in range(board_len)] for j in range(board_len)]
    for i in range(board_len):
        for j in range(board_len):
            n=s=w=e=util[i][j]
            if i>0:
                n=util[i-1][j]
            if j>0:
                w=util[i][j-1]
            if i<board_len-1:
                s=util[i+1][j]
            if j<board_len-1:
                e=util[i][j+1]
            left=0.7*w+0.1*(n+s+e)
            right=0.7*e+0.1*(n+s+w)
            up=0.7*n+0.1*(w+s+e)
            down=0.7*s+0.1*(n+w+e) 
                
            maxi=max(left,right,up,down)
            
            if maxi==up:
                policy[i][j]=("n")
            elif maxi==down:
                policy[i][j]=("s")
            elif maxi==right:
                policy[i][j]=("e")
            else:
                policy[i][j]=("w")
    return policy
    
def sequence(board_len,car,mapper,swerve,left):
    reward=car.reward
    policy=car.policy
    i,j=car.start
    t1,t2=car.end
    score=0
    k=0
    if (i,j)==(t1,t2):
        return np.float64(100)
    while (i,j)!=(t1,t2):
      
        direction=policy[i][j]
        
        if swerve[k]>0.7:
            if swerve[k]>0.8:
                if swerve[k]>0.9:
                     a,b=mapper[(i,j,left[left[direction]])] 
                else:
                    a,b=mapper[(i,j,left[left[left[direction]]])] 
            else:
                a,b=mapper[(i,j,left[direction])] 
        else:
            a,b=mapper[(i,j,direction)]
        if a>=0 and b>=0 and a<board_len and b<board_len:
            score+=reward[i][j]
            i,j=a,b
        else:
            break
        k+=1
        if k==1000000:
           k=0
        
    return np.float64(score+reward[i][j]+1)


#Main Function
start=time.time()
board_len,board,cars=parse_input()
gamma=0.9
err=0.1
delta=err*(1-gamma)/gamma
leftist={"n":"w","w":"s","s":"e","e":"n"}
opfile="output.txt"
opf=open(opfile,"w")
mapper=mapping(board_len)
randomiser=[]
for i in range(10):
    np.random.seed(i)
    randomiser.append(np.random.random_sample(1000000))

for car in cars:
    if car.start==car.end:
        car.mean_gain=100
        opf.write(str(car.mean_gain)+'\n')
        continue
    reward=reward_gen([x[:] for x in board],car)
    car.reward=reward
    utility=utility_gen(board_len,car,gamma,delta)
    car.utility=utility
    policy=policy_gen(board_len,utility)
    car.policy=policy

    for i in range(10):
        swerve=randomiser[i]
        gain=sequence(board_len,car,mapper,swerve,leftist)
        car.gain[i]=gain
    car.mean_gain=int(np.floor(sum(car.gain)/10))
    opf.write(str(car.mean_gain)+'\n')
    print(car.mean_gain)
end=time.time()
print(end-start)


opf.close()
