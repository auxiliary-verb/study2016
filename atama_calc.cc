
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <map>
#include <fstream>
#include <string>
#include <array>
#include <set>
#include <iomanip>
#include <stack>

using namespace std;

using paiarray = array<int,38>;
map<int, vector<int>> map_; 

void init(){
	int key;
	int pattern1_m;
	int pattern1_t;
	int pattern2_m;
	int pattern2_t;
	int index=1,indexloop=0;
	vector<int> tmp;

	ifstream ifs("./syanten.dat");
	string str;
	if(ifs.fail()) {
		exit(0);
	}

	while(getline(ifs, str)) {
		sscanf(str.data(),"%d %d %d %d %d", &key,&pattern1_m,&pattern1_t,&pattern2_m,&pattern2_t);
		tmp.push_back(pattern1_m);
		tmp.push_back(pattern1_t);
		tmp.push_back(pattern2_m);
		tmp.push_back(pattern2_t);
		map_[key] = tmp;
		tmp.clear();
	}

}

int checkNormalSyanten(const paiarray& tehai)
{
	int key=0;
	int pt1m=0,pt1t=0;
	int pt2m=0,pt2t=0;
	int pt=0,ptm=0,ptt=0;
	vector<int> tmp;

	for(int j=0;j<3;j++){
		key=0;
		for(int i=1+j*10,offset=100000000;i<=9+j*10;i++,offset/=10){
			key+=tehai[i]*offset;
		}

		tmp=map_[key];

		pt1m=tmp[0];
		pt1t=tmp[1];

		pt2m=tmp[2];
		pt2t=tmp[3];

		if(pt1m*2+pt1t>=pt2m*2+pt2t){
			ptm+=pt1m;
			ptt+=pt1t;
		}
		else{
			ptm+=pt2m;
			ptt+=pt2t;
		}
	}

	for(int i=31;i<=37;i++){
		if(tehai[i]>=3){
			ptm++;
		}
		else if(tehai[i]>=2){
			ptt++;
		}
	}
	if(ptm+ptt>4){
		while(ptm+ptt>4) ptt--;
	}
	
	return 8-ptm*2-ptt;
}

int NormalSyanten(paiarray tehai)
{
	int atamaresult=99;
	int result=99;
	int tmpresult=0;
	for(int i=1;i<38;i++)
	{
		if(2 <= tehai[i])
		{            
			tehai[i] -= 2;
			tmpresult=checkNormalSyanten(tehai)-1;
			if(tmpresult < atamaresult){
				atamaresult=tmpresult;
			}
			tehai[i] += 2;
		}
	}
	
	//if(fuurosuu == 0){
        tmpresult=checkNormalSyanten(tehai);
		if(tmpresult < result){
			result=tmpresult;
		}
	//}


	return atamaresult<result;
}


int calc_key(const paiarray& n) { //,vector<int> pos){
	int p = -1;
	int x = 0;
	//int pos_p = 0;
	bool b = false;
	// supai
	for(int i=0;i<3;++i){
		for(int j=0;j<9;++j){
			if(n[i*10+j+1] == 0){
				if(b){
					b=false;
					x |= 0x1 << p;
					++p;
				}
			}else{
				++p;
				b=true;
				//pos[pos_p++] = i*9+j;
				switch(n[i*10+j+1]){
				case 2:
					x|=0x3<<p;
					p += 2;
					break;
				case 3:
					x|=0xF<<p;
					p += 4;
					break;
				case 4:
					x|=0x3F<<p;
					p += 6;
					break;
				}
			}
		}
		if(b){
			b=false;
			x|=0x1<<p;
			++p;
		}
	}
	// jipai
	for(int i=31;i<38;++i){
		if(n[i]>0){
			++p;
			//pos[pos_p++] = i;
			switch(n[i]){
			case 2:
				x|=0x3<<p;
				p+=2;
				break;
			case 3:
				x|=0xF<<p;
				p+=4;
				break;
			case 4:
				x|=0x3F<<p;
				p+=6;
				break;
			}
			x|=0x1<<p;
			++p;
		}
	}
	return x;
}
/*
//template <size_t paicnt,int pos,int nextpos = ((pos+1)%10)?(pos+1):(pos+2)>
void func(std::set<int>& atama,std::set<int>& nonatama,paiarray& n,size_t paicnt,int pos){
	if(paicnt>0 || pos<38){
		int nextpos = ((pos+1)%10)?(pos+1):(pos+2);
		if(paicnt>=4){
			n[pos]=4;
			func(atama,nonatama,n,paicnt-4,nextpos);
		}
		if(paicnt>=3){
			n[pos]=3;
			func(atama,nonatama,n,paicnt-3,nextpos);
		}
		if(paicnt>=2){
			n[pos]=2;
			func(atama,nonatama,n,paicnt-2,nextpos);
		}
		if(paicnt>=1){
			n[pos]=1;
			func(atama,nonatama,n,paicnt-1,nextpos);
		}
		n[pos]=0;
		func(atama,nonatama,n,paicnt,nextpos);
	}else if(paicnt>0){
		int key = calc_key(n);
		if(NormalSyanten(n)){
			atama.insert(key);
		}else{
			nonatama.insert(key);
		}
	}
}//*/

int main(){
	init();
	std::set<int> atama;
	std::set<int> nonatama;
	paiarray n={};
	//func(atama,nonatama,n,14,1);
	
	//*
	for(size_t i=1;i<38; i+=((i+1)%10)?1:2){
		size_t paicnt = 14;
		int pos = i;
		stack<int> st;
		st.push(pos);
		n[pos]=4;
		paicnt -= 4;
		pos +=((pos+1)%10)?1:2;
		while(!(st.empty())){
			if(pos<38 && paicnt>0){
				if(n[pos]==0){
					st.push(pos);
					n[st.top()]=paicnt<4?paicnt:4;
					paicnt-=n[st.top()];
				}
				else{
					--n[st.top()];
					++paicnt;
					if(!n[st.top()]){
						st.pop();
					}
				}
				pos +=((pos+1)%10)?1:2;
			}else if (paicnt<=0){
				int key = calc_key(n);
				if(NormalSyanten(n)){
					atama.insert(key);
				}else{
					nonatama.insert(key);
				}
				pos = st.top();
			}else {
				pos = st.top();
			}
		}

	}//*/
	
	cerr<<atama.size()<<endl;
	cerr<<nonatama.size()<<endl;
	std::set<int>& out = atama.size()<nonatama.size()?atama:nonatama;
	cout<<hex<<boolalpha;
	for(auto& x:out){
		cout<<"case "<<x<<":"<<endl;
	}
	cout<<"\treturn "<<(atama.size()<nonatama.size())<<";"<<endl;
}

