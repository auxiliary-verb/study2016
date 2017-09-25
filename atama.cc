#include <array>
#include <map>
#include <vector>
#include <boost/python.hpp>
using std::map;
using std::array;
using std::vector;
using paiarray = array<int,34>;

class agariIndex{
public:
	enum
	{
		MAN = 0,
		MAN1 = 0,
		MAN2,
		MAN3,
		MAN4,
		MAN5,
		MAN6,
		MAN7,
		MAN8,
		MAN9,
		PIN = 9,
		PIN1 = 9,
		PIN2,
		PIN3,
		PIN4,
		PIN5,
		PIN6,
		PIN7,
		PIN8,
		PIN9,
		SOU = 18,
		SOU1 = 18,
		SOU2,
		SOU3,
		SOU4,
		SOU5,
		SOU6,
		SOU7,
		SOU8,
		SOU9,
		TON=27,
		NANN,
		SHA,
		PEI,
		HAK,
		HAT,
		CHU,
		KIND_OF_PAI
	};
	
	static inline int calc_key(const paiarray& n) { //,vector<int> pos){
		int p = -1;
		int x = 0;
		//int pos_p = 0;
		bool b = false;
		// supai
		for(int i=0;i<3;++i){
			for(int j=0;j<9;++j){
				if(n[i*9+j] == 0){
					if(b){
						b=false;
						x |= 0x1 << p;
						++p;
					}
				}else{
					++p;
					b=true;
					//pos[pos_p++] = i*9+j;
					switch(n[i*9+j]){
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
		for(int i=TON;i<KIND_OF_PAI;++i){
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
	static inline bool agari(int key) {
		switch(key){
			return true;
		default:
			return false;
		}
	}
	
private:
	
};

inline bool check_agari(const boost::python::list& pylist){
	using namespace boost::python;
	paiarray tepai;
	for(size_t i=0;i<9;++i){
		tepai[i+0 ]=extract<int>(pylist[i+1]);
		tepai[i+9 ]=extract<int>(pylist[i+11]);
		tepai[i+18]=extract<int>(pylist[i+21]);
	}
	for(size_t i=0;i<7;++i){
		tepai[i+27]=extract<int>(pylist[i+31]);
	}
	return agariIndex::agari(agariIndex::calc_key(tepai));
}
/*
inline bool check_tenpai(const paiarray&& tepai){
	/*
	int cnt=0;
	for(){
		cnt+=x;
	}//*//*
	
	for(auto&& x:tepai){
		++x;
		if(x<=4){
			if(agariIndex::agari(agariIndex::calc_key(tepai))){
				return true;
			}
		}
		--x;
	}
	return false;
}
//*/
BOOST_PYTHON_MODULE(agari){
	using namespace boost::python;
	def("check_agari",check_agari);

}
