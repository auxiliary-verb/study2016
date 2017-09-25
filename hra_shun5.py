# coding: utf-8
import numpy as np

import sys

import chainer
from chainer import cuda, Function, Variable, optimizers, serializers
from chainer import Link, Chain, ChainList

import cupy as xp
import chainer.functions as F
import chainer.links as L

import syanten
import agari

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

# 
KIND_OF_HEADS = 3*7 + 1


# 牌配列のサイズ
PAI_SIZE = 38

# 過去何コマを見るか
STATE_NUM = 1

# 入力ノード 数   手牌              捨て牌              = 68
INPUT_NODE = (  KIND_OF_PAI +   KIND_OF_PAI ) * 4    * STATE_NUM

# 出力ノード数
OUTPUT_NODE =   KIND_OF_PAI

tehai_num = 13

test_highscore = 0

OUTPUT_FRAME = 1000

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

def score(index,sim,loop=100):
    global test_highscore
    total_reward = 0.
    if loop > 0:
        for i in range(loop):
            total_reward +=sim.run(train=False)
        total_reward/=loop
        if test_highscore<total_reward:
            print "highscore!",
            #serializers.save_npz('model/%06d_hs.model'%i, agent.model)
            test_highscore=total_reward
    print
    if loop > 0:
        print total_reward,
    print "epsilon:%2.2e" % agent.get_epsilon(),
    print "loss:%2.2e" % agent.outloss
    agent.outloss = 0
    #aw=agent.total_reward_award
    #print "min:%d,max:%d" % (np.min(aw),np.max(aw))
    fvs=open("value_shun.csv","a")
    fvs.write("\n");
    fvs.close()
    out="%d,%2.2e,%2.2e,%2.2e\n" % (index,total_reward,agent.get_epsilon(),agent.outloss)
    fw.write(out)
    fw.flush()

# DQN内部で使われるニューラルネット
class HRA(Chain):
    def __init__(self,hidden = 10,state_num=INPUT_NODE):
        super(HRA,self).__init__(
             l1=L.Linear(state_num, hidden),  # stateがインプット
             #l2=L.Linear(7, 64), # 8192
             #l3=L.Linear(64, 8), # 16384 -> 32768
             #l4=L.Linear(256, 64), # 8192
             #l5=L.Linear(64, 16), # 16384 -> 32768
             l_out=L.Linear(hidden, OUTPUT_NODE), # 出力2チャネル(Qvalue)がアウトプット
        )
    
    '''
    def  __call__(self,x):
        h1 = F.leaky_relu(self.l1(x))
        h2 = F.leaky_relu(self.l2(h1))
        h3 = F.leaky_relu(self.l3(h2))
        y =  F.leaky_relu(self.l4(h3))
        return y
    '''
    def __call__(self,x,t,ratio=0.5):
        return F.mean_squared_error(self.predict(x,train=True,ratio = ratio),t)

    def  predict(self,x,train=False, ratio = 0.5):
        #h1 = F.dropout(F.leaky_relu(self.l1(x)),train = train, ratio = ratio)
        #h2 = F.dropout(F.leaky_relu(self.l2(h1)),train = train, ratio = ratio)
        #h3 = F.dropout(F.leaky_relu(self.l3(h2)),train = train, ratio = ratio)
        h = F.relu(self.l1(x))
        #h = F.relu(self.l2(h))
        #h = F.relu(self.l3(h))
        #h = F.leaky_relu(self.l4(h))
        #h = F.leaky_relu(self.l5(h))
        #h = F.leaky_relu(self.l6(h))
        y =  self.l_out(h)
        return y
    #'''

# DQN内部で使われるニューラルネット
class Q:
    def __init__(self,state_num=INPUT_NODE):
        self.heads = []
        for i in range( KIND_OF_HEADS -1 ):
            model = HRA()
            optimizer = optimizers.Adam()
            optimizer.setup(model)
            self.heads.append((model,optimizer))
        model = HRA(hidden = 21+34)
        optimizer = optimizers.Adam()
        optimizer.setup(model)
        self.heads.append((model,optimizer))

    
    '''
    def  __call__(self,x):
        h1 = F.leaky_relu(self.l1(x))
        h2 = F.leaky_relu(self.l2(h1))
        h3 = F.leaky_relu(self.l3(h2))
        y =  F.leaky_relu(self.l4(h3))
        return y
    '''

    
    def predict(self,x,train=False, ratio = 0.5):
        y = xp.zeros(OUTPUT_NODE)
        for head in self.heads:
            y += head[0].predict(x,train,ratio).data[0].copy()/KIND_OF_HEADS
        return y
    #'''

# DQNアルゴリズムにしたがって動作するエージェント
class DQNAgent():
    def __init__(self, args, epsilon=1.00):
        self.model = Q()
        #model = L.Classifier(MLP(784, args.unit, 10))
        if args.gpu >= 0:
            chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
            for head in self.model.heads:
                head[0].to_gpu()  # Copy the model to the GPU

        
        self.epsilon = epsilon # ランダムアクションを選ぶ確率
        self.actions=np.array(range(KIND_OF_PAI)) #　行動の選択肢
        self.experienceMemory = [] # 経験メモリ
        self.memSize = 27*10000  # 経験メモリのサイズ(300サンプリングx100エピソード)
        self.experienceMemory_local=[] # 経験メモリ（エピソードローカル）
        self.memPos = 0 #メモリのインデックス
        
        self.batch_num = 108 # 学習に使うバッチサイズ
        
        self.gamma = 0.9       # 割引率
        self.loss=0
        self.outloss = 0

        self.monte_size = 4

        #self.total_reward_award=np.ones(100)*-1000 #100エピソード

    def get_action_value(self, seq,model, train = False,ratio = 5.0):
        # seq後の行動価値を返す
        x = Variable(cuda.to_gpu(np.hstack([seq]).astype(np.float32).reshape((1,-1))))
        out = model.predict(x=x,train = train,ratio = ratio)
        if not isinstance(out,xp.ndarray):
            out = out.data[0].copy()
        fvs = 0
        if train:
            fvs=open("value_shun.csv","a")
        for i in range(OUTPUT_NODE):
            if seq[i]<1.:
                out[i] = -float("inf") # 選ばれちゃいけない系の選択肢は潰す

            if train:
                fvs.write(str(out[i]) + ",");
        #print out
        if train:
            fvs.write("\n");
            fvs.close()
        return out

    def get_greedy_action(self, seq):
        action_index = np.argmax(self.get_action_value(seq,model = self.model))
        return self.actions[int(action_index)]

    def reduce_epsilon(self):
        pass
        #self.epsilon-=1.0/50000
        #if self.epsilon<0.05:
        #    self.epsilon = 0.05

    def get_epsilon(self):
        return self.epsilon

    def get_action(self,seq,train):
        '''
        seq (theta, old_theta)に対して
        アクション（モータのトルク）を返す。
        '''
        action=0
        if train==True and np.random.random()<self.epsilon:
            # random
            dummyactions = []
            for i in range(KIND_OF_PAI):
                if seq[i]>=1.:
                    dummyactions.append(i) # 選ばれてほしい系の選択肢をのこす
            assert len(dummyactions)>0, "error seq"
            #print dummyactions
            action = np.random.choice(np.array(dummyactions))
        else:
            # greedy
            action= self.get_greedy_action(seq)
        return action

    def experience_local(self,old_seq, action, reward, endflag, now_seq):
        #エピソードローカルな記憶
        #np.append(old_seq,actioin),
        #self.experienceMemory_local
        #print len(reward)
        self.experienceMemory_local.append( np.hstack([old_seq,action,reward[:],endflag,now_seq]) )

    def experience_global(self):
        """
        #グローバルな記憶
        #ベスト100に入る経験を取り込む
        if np.min(self.total_reward_award)<total_reward:
            i=np.argmin(self.total_reward_award)
            self.total_reward_award[i]=total_reward

            # GOOD EXPERIENCE REPLAY
            for x in self.experienceMemory_local:
                self.experience( x )

        #一定確率で優秀でないものも取り込む
        if np.random.random()<0.01:
            # # NORMAL EXPERIENCE REPLAY
            for x in self.experienceMemory_local:
                self.experience( x )
        """
        for x in self.experienceMemory_local:
            self.experience( x )
        self.experienceMemory_local=[]

    def experience(self,x):
        if len(self.experienceMemory)>self.memSize:
            self.experienceMemory[int(self.memPos%self.memSize)]=x
            self.memPos+=1
        else:
            self.experienceMemory.append( x )

    def conv_state(self,dtehai,dhou):
        # 一人麻雀なら状態は手牌+河で 手牌は本来順序によって変化するべき
        out = np.zeros(INPUT_NODE)
        for j in range(len(dtehai)):
            if dtehai[j]>0:
                out[j] = 1
            if dtehai[j]>1:
                out[j + (KIND_OF_PAI +   KIND_OF_PAI)*1] = 1
            if dtehai[j]>2:
                out[j + (KIND_OF_PAI +   KIND_OF_PAI)*2] = 1
            if dtehai[j]>3:
                out[j + (KIND_OF_PAI +   KIND_OF_PAI)*3] = 1

            if dhou[j]>0:
                out[j + KIND_OF_PAI] = 1
            if dhou[j]>1:
                out[j + KIND_OF_PAI + (KIND_OF_PAI +   KIND_OF_PAI)*1] = 1
            if dhou[j]>2:
                out[j + KIND_OF_PAI + (KIND_OF_PAI +   KIND_OF_PAI)*2] = 1
            if dhou[j]>3:
                out[j + KIND_OF_PAI + (KIND_OF_PAI +   KIND_OF_PAI)*3] = 1
        return out 
    def update_model(self):
        '''
        モデルを更新する
        '''
        # 経験メモリにたまってない場合は更新しない
        if len(self.experienceMemory)<self.batch_num:
            return

        # 経験メモリからバッチを作成
        memsize=len(self.experienceMemory)
        batch_index = list(np.random.randint(0,memsize,(self.batch_num)))
        batch =np.array( [self.experienceMemory[i] for i in batch_index ])
        x = Variable(cuda.to_gpu(batch[:,0:INPUT_NODE].reshape( (self.batch_num,-1)).astype(np.float32)))
        

        for hindex in range(KIND_OF_HEADS):
            targets=self.model.heads[hindex][0].predict(x).data.copy()
            for i in range(self.batch_num):
                #[ seq..., action, reward, seq_new]
                #print len(batch[i])
                a = batch[i,INPUT_NODE]
                r = batch[i, INPUT_NODE + 1 + hindex]
                end = batch[i, INPUT_NODE + 1 + KIND_OF_HEADS]
                if end==0:
                    new_seq= batch[i,(INPUT_NODE + 2 + KIND_OF_HEADS):( INPUT_NODE*2 + 2 + KIND_OF_HEADS)]
                    #print new_seq.size==(INPUT_NODE + 2 + KIND_OF_HEADS)
                    targets[i,PAI_ACT2NUM[int(a)]]=( r+ self.gamma * np.max(self.get_action_value(new_seq,model = self.model.heads[hindex][0])))
                    '''
                    dummy_seq = batch[i,(INPUT_NODE + 3):( INPUT_NODE + KIND_OF_PAI*2 + 3)]
                    random_pai = []
                    for pai_index in range(len(PAI_NUM2ACT)):
                        for j in range(int(4-dummy_seq[pai_index]-dummy_seq[pai_index+KIND_OF_PAI])):
                            random_pai.append(PAI_NUM2ACT[pai_index])
                    np.random.shuffle(random_pai)
                    if len(random_pai)>self.monte_size:
                        random_pai = random_pai[0:self.monte_size] #important##########################################################################
                    for pai in random_pai:
                        dummy_seq[pai]+=1
                        dummy_new_seq = self.conv_state(dummy_seq[0:KIND_OF_PAI], dummy_seq[KIND_OF_PAI:KIND_OF_PAI*2])
                        next_Q += np.max(self.get_action_value(dummy_new_seq))/len(random_pai)
                        dummy_seq[pai]-=1
                    
                    if a!=0:
                        targets[i,PAI_ACT2NUM[int(a)]]=( r+ self.gamma * next_Q)
                    else:
                        assert "update_model() a=0"
                        #targets[i,PAI_ACT2NUM[int(a)]]=( r )
                    '''

                else:
                    #"""
                    if a!=0:
                        targets[i,PAI_ACT2NUM[int(a)]]=( r )
                    else:
                        for pai in range(KIND_OF_PAI):
                            targets[i,pai]=  ( r )
                    #"""
                #if PAI_ACT2NUM[int(a)]>=34:
                #    print str(i)+","+str(int(a))
                
            t = Variable(xp.array(targets).reshape((self.batch_num,-1)).astype(xp.float32))

            # ネットの更新
            self.model.heads[hindex][0].zerograds()
            loss=self.model.heads[hindex][0](x ,t)
            self.loss = loss.data
            self.outloss += loss.data/OUTPUT_FRAME
            loss.backward()
            self.model.heads[hindex][1].update()

class pendulumEnvironment():
    '''
    麻雀の環境
    '''
    def __init__(self):
        self.pais = np.zeros(KIND_OF_PAI*4,dtype=int)
        for i in range(9):
            self.pais[i*4  ] = i+1
            self.pais[i*4+1] = i+1
            self.pais[i*4+2] = i+1
            self.pais[i*4+3] = i+1
            self.pais[(i+9)*4  ] = i + 11
            self.pais[(i+9)*4+1] = i + 11
            self.pais[(i+9)*4+2] = i + 11
            self.pais[(i+9)*4+3] = i + 11
            self.pais[(i+18)*4  ] = i + 21
            self.pais[(i+18)*4+1] = i + 21
            self.pais[(i+18)*4+2] = i + 21
            self.pais[(i+18)*4+3] = i + 21
        for i in range(7):
            self.pais[(i+27)*4  ] = i + 31
            self.pais[(i+27)*4+1] = i + 31
            self.pais[(i+27)*4+2] = i + 31
            self.pais[(i+27)*4+3] = i + 31
        self.tehai = np.zeros(PAI_SIZE,dtype=int) # 手牌 四人ならこれがあと4ついるはず
        self.hou = np.zeros(PAI_SIZE,dtype=int) # 河の牌
        self.sya = syanten.Syanten()

        self.atama = 0

        self.reset(0,0)
        self.syanten = self.get_syanten()
        self.shuntu = np.zeros(PAI_SIZE)

    def get_shuntu(self,first_pai):
        if self.tehai[first_pai] > 3 and self.tehai[first_pai+1] > 3 and self.tehai[first_pai+2] > 3:
            return 4
        elif self.tehai[first_pai] > 2 and self.tehai[first_pai+1] > 2 and self.tehai[first_pai+2] > 2:
            return 3
        elif self.tehai[first_pai] > 1 and self.tehai[first_pai+1] > 1 and self.tehai[first_pai+2] > 1:
            return 2
        elif self.tehai[first_pai] > 0 and self.tehai[first_pai+1] > 0 and self.tehai[first_pai+2] > 0:
            return 1
        return 0
    def get_shuntu_cnt(self):
        out = 0
        dummy = self.tehai[:]
        for p in PAI_NUM2ACT:
            if 30<= p:
                break
            if p%10<8:
                x = self.get_shuntu(p)
                if x>0:
                    dummy[p  ]-=x
                    dummy[p+1]-=x
                    dummy[p+2]-=x
                    out += x
        return out


    def set_shuntu(self):
        for p in PAI_NUM2ACT:
            if p<8:
                self.shuntu[p] = self.get_shuntu(p)

    def get_shuntu_reward(self,first_pai):
        oldshuntu = self.shuntu[first_pai]
        self.shuntu[first_pai] = self.get_shuntu(first_pai)
        return self.shuntu[first_pai] - oldshuntu

    def get_atama(self):
        return self.sya.check_atama()


    def get_atama_reward(self):
        oldatama = self.atama
        self.atama = self.get_atama()
        return self.atama - oldatama

    def get_syanten(self):
        '''
        シャンテン数を返す 値次第で和了
        '''
        #self.sya.set_tehai(self.tehai.tolist())#.tolist()
        return min(self.sya.NormalSyanten(), self.sya.KokusiSyanten(), self.sya.TiitoituSyanten())
    
    def set_sya_tehai(self):
        self.sya.set_tehai(self.tehai.tolist())#.tolist()

    def check_agari(self):
        return agari.check_agari(self.tehai.tolist())
    
    def reset(self,initial_theta, initial_dtheta):
        #self.th          = initial_theta
        #self.th_old   = self.th
        #self.th_ = initial_dtheta
        #self.g=0.01
        #self.highscore=-1.0
        for i in range(PAI_SIZE):
            self.tehai[i] = 0
            self.hou[i] = 0
        np.random.shuffle(self.pais)
        for i in range(tehai_num):
            index = self.pais[i]
            #print int(index)
            self.tehai[int(index)] += 1
        self.pais_position = tehai_num
        self.set_sya_tehai()
        self.syanten = self.get_syanten()
        self.get_atama_reward()

    def get_reward(self):
        '''
        高さプラスなら5倍ボーナスで高さに比例した正の報酬
        マイナスなら低さに比例した負の報酬
        '''
        '''
        reward=0
        h=-np.cos(self.th)
        if h>=0:
            reward= 5*np.abs(h)
        else:
            reward= -np.abs(h)
        return reward
        '''
        oldsyanten = self.syanten
        self.syanten = self.get_syanten()
        return oldsyanten - self.syanten

    def get_state(self):
        # 一人麻雀なら状態は手牌+河で 手牌は本来順序によって変化するべき
        dtehai = pais2act(self.tehai)
        dhou = pais2act(self.hou)
        out = np.zeros(INPUT_NODE)
        for j in range(len(dtehai)):
            if dtehai[j]>0:
                out[j] = 1
            if dtehai[j]>1:
                out[j + (KIND_OF_PAI +   KIND_OF_PAI)*1] = 1
            if dtehai[j]>2:
                out[j + (KIND_OF_PAI +   KIND_OF_PAI)*2] = 1
            if dtehai[j]>3:
                out[j + (KIND_OF_PAI +   KIND_OF_PAI)*3] = 1

            if dhou[j]>0:
                out[j + KIND_OF_PAI] = 1
            if dhou[j]>1:
                out[j + KIND_OF_PAI + (KIND_OF_PAI +   KIND_OF_PAI)*1] = 1
            if dhou[j]>2:
                out[j + KIND_OF_PAI + (KIND_OF_PAI +   KIND_OF_PAI)*2] = 1
            if dhou[j]>3:
                out[j + KIND_OF_PAI + (KIND_OF_PAI +   KIND_OF_PAI)*3] = 1
        return out 
    
    def get_tehai(self):
        return pais2act(self.tehai)
    def get_hou(self):
        return pais2act(self.hou)

    def update_state(self, action):
        '''
        action はモータのトルク。符号しか意味を持たない。
        正なら0.005, 0なら0, 負なら-0.005
        '''
        '''
        power = 0.005* np.sign(action)
        self.th_ += -self.g*np.sin(self.th)+power
        self.th_old = self.th
        self.th += self.th_
        '''

        #print self.tehai
        #print self.hou
        self.tehai[int(action)] -= 1
        self.hou[int(action)] += 1
        #print self.tehai
        #print self.hou
        for x in self.tehai:
            assert x>-1, "error tehai {}".format(self.tehai)
        for x in self.hou:
            assert x<5, "error hou {}".format(self.hou)
    def tumo(self):
        self.tehai[self.pais[self.pais_position]] += 1
        self.pais_position += 1

    def get_svg(self):
        """
        アニメーション用に現在の状況をSVGで返す
        """
        dr=sw.Drawing("hoge.svg",(150,150))
        c=(75,75)
        dr.add(dr.line(c,(c[0]+50*np.sin(self.th),c[1]+50*np.cos(self.th)), stroke=sw.utils.rgb(0,0,0),stroke_width=3))
        return SVG(dr.tostring())

# 環境とエージェントを渡すとシミュレーションするシミュレータ。
# ここにシーケンスを持たせるのはなんか変な気もするけどまあいいか。。
class simulator:
    def __init__(self, environment, agent):
        self.agent = agent
        self.env = environment

        self.num_seq=INPUT_NODE
        self.reset_seq()
        self.learning_rate=1.0
        self.highscore=0
        self.log=[]

    def reset_seq(self):
        self.seq=np.zeros(self.num_seq)

    def push_seq(self, state):
        for i in range(self.seq.size-state.size):
            self.seq[self.seq.size-i]=self.seq[self.seq.size-state.size-i]
        for i in range(state.size):
            self.seq[i]=state[i]
        #self.seq = (np.append(state,self.seq))[0:self.num_seq]
        #self.seq[INPUT_NODE:self.num_seq]=self.seq[0:self.num_seq-INPUT_NODE]

    def run(self, train=True, movie=False, enableLog=False):
        agari = 0
        self.env.reset(0,0)

        self.reset_seq()
        total_reward=0
        
        # init seq
        #state = self.env.get_state()
        #self.push_seq(state)
        flag = False
        #ssyanten = self.env.get_syanten()
        # 一人麻雀では自摸回数は27回
        self.env.set_sya_tehai()
        reward = self.env.get_reward()
        self.env.get_atama_reward()
        self.env.set_shuntu()
        for i in range(27):
            # 麻雀の特性上エージェントの一人は行動前に環境が変化する
            self.env.tumo()
            state = self.env.get_state()
            self.push_seq(state)
            self.env.set_sya_tehai()
            reward = []
            for p in PAI_NUM2ACT:
                if 30<= p:
                    break
                if p%10<8:
                    reward.append(self.env.get_shuntu_reward(p))
            reward.append(self.env.get_atama_reward())
            

            # total_reward += reward
            # エピソードローカルなメモリに記憶する
            if train and flag:
                self.agent.experience_local(old_seq, action, reward, 0, self.seq.copy())    

            # 現在のstateからなるシーケンスを保存
            old_seq = self.seq.copy()
            #if movie:
            #    self.agent.get_action_value(old_seq,train = True,model = self.model)
            agari = self.env.check_agari()
            action = 0
            if agari:
                #agari = 1
                #reward = 1
                # エピソードローカルなメモリに記憶する
                #if train:
                #    self.agent.experience_local(old_seq, action, reward, 1, old_seq)
                break
            flag = True
            # エージェントの行動を決める
            action = act2PaiNumber(self.agent.get_action(old_seq,train))

            # 環境に行動を入力する
            self.env.update_state(action)
            #reward=self.env.get_reward()
            #total_reward +=reward

            # 結果を観測してstateとシーケンスを更新する
            #state = self.env.get_state()
            #self.push_seq(state)
            #new_seq = self.seq.copy()

            
            if enableLog:
                self.log.append(np.hstack([old_seq[0], action, reward]))
            '''
            # 必要ならアニメを表示する
            if movie:
                display.clear_output(wait=True)
                display.display(self.env.get_svg())
                time.sleep(0.01)
            '''
        '''
        if train and flag and self.env.get_syanten() != -1:
                self.agent.experience_local(old_seq, action, 0, 1, old_seq)
        '''
        if train :
            #syanten = self.env.get_syanten()
            #self.agent.experience_local(old_seq, action, 2./(1.+math.exp(-syanten))-1., 1, old_seq)
            self.agent.experience_local(old_seq, action, reward, 1, old_seq)

        if train:
            # エピソードローカルなメモリ内容をグローバルなメモリに移す
            self.agent.experience_global()
            # 学習用メモリを使ってモデルを更新する
            self.agent.update_model()
            self.agent.reduce_epsilon()

        if enableLog:
            return agari,self.log
        return 1 if agari else 0
        #return total_reward


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    

    #global test_highscore
    agent=DQNAgent(args)
    env=pendulumEnvironment()
    sim=simulator(env,agent)

    test_highscore=0

    fw=open("value_shun.csv","w")
    fw.close()

    fw=open("log.csv","w")
    tumo_cnt = 0


    for i in xrange(30000000):
        '''
        sys.stdout.write("\rtest%d"%i)
        sys.stdout.flush()
        '''
        #print "test%d"%i
        total_reward=sim.run(train=True, movie=False)
        tumo_cnt += total_reward
        """
        if i%1000 ==0:
            serializers.save_npz('model/%06d.model'%i, agent.model)
        """
        
        if i%OUTPUT_FRAME == 0:
            print "test%d"%i
            if i%(OUTPUT_FRAME*10)==0 and i!=0:
                score(i,sim,1000)
            else:
                score(i,sim,0)
            print tumo_cnt
            tumo_cnt=0
    print "test30000000"
    score(i,sim,1000000)
    print tumo_cnt
    fw.close
