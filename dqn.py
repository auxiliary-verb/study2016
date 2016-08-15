# coding: utf-8
import numpy as np

import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

import syanten

np.random.seed(0)

# 牌の種類数
KIND_OF_PAI = 9*3+4+3

# 牌配列のサイズ
PAI_SIZE = 38

# 過去何コマを見るか
STATE_NUM = 1

# 入力ノード 数   手牌              捨て牌         = 68
INPUT_NODE = (  KIND_OF_PAI +   KIND_OF_PAI     ) * STATE_NUM

# 出力ノード数
OUTPUT_NODE =   KIND_OF_PAI

# DQN内部で使われるニューラルネット
class Q(Chain):
    def __init__(self,state_num=INPUT_NODE*STATE_NUM):
        super(Q,self).__init__(
             l1=L.Linear(state_num, 1024),  # stateがインプット
             l2=L.Linear(1024, 512),
             l3=L.Linear(512, 256),
             l4=L.Linear(256, 128),
             l5=L.Linear(128, OUTPUT_NODE), # 出力2チャネル(Qvalue)がアウトプット
        )

    def __call__(self,x,t):
        return F.mean_squared_error(self.predict(x,train=True),t)

    def  predict(self,x,train=False):
        h1 = F.leaky_relu(self.l1(x))
        h2 = F.leaky_relu(self.l2(h1))
        h3 = F.leaky_relu(self.l3(h2))
        h4 =  F.leaky_relu(self.l4(h3))
        y = F.leaky_relu(self.l5(h4))
        return y

# DQNアルゴリズムにしたがって動作するエージェント
class DQNAgent():
    def __init__(self, epsilon=0.99):
        self.model = Q()
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)
        self.epsilon = epsilon # ランダムアクションを選ぶ確率
        self.actions=range(KIND_OF_PAI) #　行動の選択肢
        self.experienceMemory = [] # 経験メモリ
        self.memSize = 300*100  # 経験メモリのサイズ(300サンプリングx100エピソード)
        self.experienceMemory_local=[] # 経験メモリ（エピソードローカル）
        self.memPos = 0 #メモリのインデックス
        self.batch_num = 32 # 学習に使うバッチサイズ
        self.gamma = 0.9       # 割引率
        self.loss=0

        #self.total_reward_award=np.ones(100)*-1000 #100エピソード

    def get_action_value(self, seq):
        # seq後の行動価値を返す
        x = Variable(np.hstack([seq]).astype(np.float32).reshape((1,-1)))
        out = self.model.predict(x).data[0]
        for i in range(KIND_OF_PAI):
            if seq[i]<1.:
                out[i] = -30. # 選ばれちゃいけない系の選択肢は潰す
        return out

    def get_greedy_action(self, seq):
        action_index = np.argmax(self.get_action_value(seq))
        return self.actions[action_index]

    def reduce_epsilon(self):
        self.epsilon-=1.0/1000000

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
            if len(dummyactions)>0:
	            action = np.random.choice(np.array(dummyactions))
        else:
            # greedy
            action= self.get_greedy_action(seq)
        return action

    def experience_local(self,old_seq, action, reward, new_seq):
        #エピソードローカルな記憶
        self.experienceMemory_local.append( np.hstack([old_seq,action,reward,new_seq]) )

    def experience_global(self,total_reward):
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

    def update_model(self,old_seq, action, reward, new_seq):
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
        x = Variable(batch[:,0:INPUT_NODE].reshape( (self.batch_num,-1)).astype(np.float32))
        targets=self.model.predict(x).data.copy()

        for i in range(self.batch_num):
            #[ seq..., action, reward, seq_new]
            a = batch[i,INPUT_NODE]
            r = batch[i, INPUT_NODE+1]
            ai=int((a+1)/2) #±1 をindex(0,1)に
            new_seq= batch[i,(INPUT_NODE+2):(INPUT_NODE*2+2)]
            targets[i,ai]=( r+ self.gamma * np.max(self.get_action_value(new_seq)))
        t = Variable(np.array(targets).reshape((self.batch_num,-1)).astype(np.float32)) 

        # ネットの更新
        self.model.zerograds()
        loss=self.model(x ,t)
        self.loss = loss.data
        loss.backward()
        self.optimizer.update()

class pendulumEnvironment():
    '''
    麻雀の環境
    '''
    def __init__(self):
        self.pais = np.zeros(KIND_OF_PAI*4)
        for i in range(9):
            self.pais[i  ] = i+1
            self.pais[i+1] = i+1
            self.pais[i+2] = i+1
            self.pais[i+3] = i+1
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
        self.tehai = np.zeros(PAI_SIZE) # 手牌 四人ならこれがあと4ついるはず
        self.hou = np.zeros(PAI_SIZE) # 河の牌
        self.sya = syanten.Syanten()
        self.reset(0,0)

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
        for i in range(13):
            self.tehai[self.pais[i]] += 1
        self.pais_position = 13

    def get_reward(self,agent):
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
        self.sya.set_tehai(self.tehai)

    def get_syanten(self):
        '''
        シャンテン数を返す 値次第で和了
        '''
        self.sya.set_tehai(self.tehai)
        out = np.zeros(3)
        out[0] = self.sya.NormalSyanten()
        out[1] = self.sya.KokusiSyanten()
        out[2] = self.sya.TiitoituSyanten()
        return np.max(out)

    def get_state(self):
        # 一人麻雀なら状態は手牌+河で 手牌は本来順序によって変化するべき
        return self.tehai + self.hou 

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
        self.tehai[action] -= 1
        self.hou[action] += 1
        for x in self.tehai:
            assert x<0, "error tehai"
        for x in self.hou:
            assert x>4, "error hou"
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
        self.seq = (np.append(state,self.seq))[0:self.num_seq]
        #self.seq[INPUT_NODE:self.num_seq]=self.seq[0:self.num_seq-INPUT_NODE]

    def run(self, train=True, movie=False, enableLog=False):

        self.env.reset(0,0)

        self.reset_seq()
        total_reward=0
        
        # init seq
        state = self.env.get_state()
        self.push_seq(state)

        # 一人麻雀では自摸回数は27回
        for i in range(27):
            # 麻雀の特性上エージェントの一人は行動前に環境が変化する
            self.env.tumo()

            # 現在のstateからなるシーケンスを保存
            old_seq = self.seq.copy()

            # エージェントの行動を決める
            action = self.agent.get_action(old_seq,train)

            # 環境に行動を入力する
            self.env.update_state(action)
            reward=self.env.get_reward()
            total_reward +=reward

            # 結果を観測してstateとシーケンスを更新する
            state = self.env.get_state()
            self.push_seq(state)
            new_seq = self.seq.copy()

            # エピソードローカルなメモリに記憶する
            self.agent.experience_local(old_seq, action, reward, new_seq)

            if enableLog:
                self.log.append(np.hstack([old_seq[0], action, reward]))
            '''
            # 必要ならアニメを表示する
            if movie:
                display.clear_output(wait=True)
                display.display(self.env.get_svg())
                time.sleep(0.01)
            '''


        # エピソードローカルなメモリ内容をグローバルなメモリに移す
        self.agent.experience_global(total_reward)

        if train:
            # 学習用メモリを使ってモデルを更新する
            self.agent.update_model(old_seq, action, reward, new_seq)
            self.agent.reduce_epsilon()

        if enableLog:
            return total_reward,self.log

        return total_reward


if __name__ == '__main__':
    agent=DQNAgent()
    env=pendulumEnvironment()
    sim=simulator(env,agent)

    test_highscore=0

    fw=open("log.csv","w")

    for i in range(30000):
        total_reward=sim.run(train=True, movie=False)

        if i%1000 ==0:
            serializers.save_npz('model/%06d.model'%i, agent.model)

        if i%10 == 0:
            total_reward=sim.run(train=False, movie=False)
            if test_highscore<total_reward:
                print "highscore!",
                serializers.save_npz('model/%06d_hs.model'%i, agent.model)
                test_highscore=total_reward
            print i,
            print total_reward,
            print "epsilon:%2.2e" % agent.get_epsilon(),
            print "loss:%2.2e" % agent.loss,
            aw=agent.total_reward_award
            print "min:%d,max:%d" % (np.min(aw),np.max(aw))

            out="%d,%d,%2.2e,%2.2e,%d,%d\n" % (i,total_reward,agent.get_epsilon(),agent.loss, np.min(aw),np.max(aw))
            fw.write(out)
            fw.flush()
    fw.close
