#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
created test_distribution.py by rjw at 19-1-10 in WHU.
"""

import numpy as np
from math import pi, exp, sqrt
# for tid2013 mos-[0-9]
def f(x, mean, var):
    var=var*var
    sigma=1e-9
    return exp(-(x-mean)**2/(2*var+sigma))/sqrt(2*pi*var+sigma)

def get_score_distribution(mean,var):
    scores_list=[]
    for i in range(10):
        scores_list.append(f(i,mean,var))
    return scores_list

def other(mean, var):
    s = np.random.normal(mean,var,10000)
    s = np.rint(s)
    a=np.histogram(s,bins=np.arange(1,12),density=True)
    return a[0]

if __name__=="__main__":

    f1=get_score_distribution(5.51429 ,0.13013)
    print(f1,type(f1))
    print("=================")
    f2=other(5.51429 ,0.13013)
    print(f2,type(f2))