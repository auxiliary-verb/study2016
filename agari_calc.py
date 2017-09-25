# coding: utf-8
import numpy as np

import sys

import cupy as xp
import chainer.functions as F
import chainer.links as L

import syanten
import agari

import time

import argparse

'''
if gpu :
    xp = cupy
else:
    xp = numpy
'''
np.random.seed(1033)# 0 1033 1631
xp.random.seed(1033)#

# 牌の種類数
KIND_OF_PAI = 9*3+4+3

# 牌配列のサイズ
PAI_SIZE = 38

# 過去何コマを見るか
STATE_NUM = 1

# 入力ノード 数   手牌              捨て牌              = 68
INPUT_NODE = (  KIND_OF_PAI +   KIND_OF_PAI ) * 4    * STATE_NUM

# 出力ノード数
OUTPUT_NODE =   KIND_OF_PAI

test_highscore = 0

OUTPUT_FRAME = 10000

PAI_NUM2ACT = [  1, 2, 3, 4, 5, 6, 7, 8, 9, \
                11,12,13,14,15,16,17,18,19, \
                21,22,23,24,25,26,27,28,29, \
                31,32,33,34,35,36,37]
PAI_ACT2NUM = [ 34, 0, 1, 2, 3, 4, 5, 6, 7, 8,\
                34, 9,10,11,12,13,14,15,16,17,\
                34,18,19,20,21,22,23,24,25,26,\
                34,27,28,29,30,31,32,33,34]
def act2PaiNumber(action):
    if action<9:
        return action+1
    elif action<18:
        return action+2
    elif action<27:
        return action+3
    elif action<34:
        return action+4

def pais2act(pais):
    act = np.zeros(KIND_OF_PAI)
    for i in range(9):
        act[i   ] = pais[i+1]
        act[i+9 ] = pais[i+11]
        act[i+18] = pais[i+21]
    for i in range(7):
        act[i+27] = pais[i+31]
    return act

if __name__ == '__main__':
    pais = np.zeros(KIND_OF_PAI*4,dtype=int)
    for i in range(9):
        pais[i*4  ] = i+1
        pais[i*4+1] = i+1
        pais[i*4+2] = i+1
        pais[i*4+3] = i+1
        pais[(i+9)*4  ] = i + 11
        pais[(i+9)*4+1] = i + 11
        pais[(i+9)*4+2] = i + 11
        pais[(i+9)*4+3] = i + 11
        pais[(i+18)*4  ] = i + 21
        pais[(i+18)*4+1] = i + 21
        pais[(i+18)*4+2] = i + 21
        pais[(i+18)*4+3] = i + 21
    for i in range(7):
        pais[(i+27)*4  ] = i + 31
        pais[(i+27)*4+1] = i + 31
        pais[(i+27)*4+2] = i + 31
        pais[(i+27)*4+3] = i + 31

    tehai = np.zeros(PAI_SIZE,dtype=int)
    sya = syanten.Syanten()
    def get_syanten():
        '''
        シャンテン数を返す 値次第で和了
        '''
        sya.set_tehai(tehai.tolist())#.tolist()
        return min(sya.NormalSyanten(), sya.KokusiSyanten(), sya.TiitoituSyanten())
    def get_agari():
        return agari.check_agari(tehai.tolist())
    def reset():
        for i in range(PAI_SIZE):
            tehai[i] = 0
        np.random.shuffle(pais)
        for i in range(14):
            index = pais[i]
            #print int(index)
            tehai[int(index)] += 1
    seido = 0
    seitou = 0
    syanten_t = 0
    agari_t = 0
    np.random.seed(1033)# 0 1033 1631
    start = time.time()
    for i in xrange(10000000):
        reset()
        syantensu = get_syanten()
        if( syantensu == -1):
            seitou = seitou+1
    syanten_t += time.time()-start
    print "syanten = " + str(syanten_t) 
    print seitou
    np.random.seed(1033)# 0 1033 1631
    start = time.time()
    for i in xrange(10000000):
        agaric = get_agari()
        if( agaric==True ):
            seitou = seitou+1
    agari_t += time.time()-start
    print "agari = " + str(agari_t)
    print seitou
