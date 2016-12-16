#include "syanten.h"



//中身クリア
void Syanten::clear()
{
	for(int i=0;i<38;i++){
		tehai[i]=0;
	}
	tehai[38]=-1;
	fuurosuu=0;
}

//手牌セット
void Syanten::set_tehai(int t[])
{
	for(int i=0;i<38;i++){
		tehai[i]=t[i];
	}
	tehai[38]=-1;
}

//国士シャンテン
int Syanten::KokusiSyanten()
{
	int kokusi_toitu=0,syanten_kokusi=13,i;             
	//老頭牌
	for(i=1;i<30;i++){        
		if(i%10==1||i%10==9||i%20==1||i%20==9){	   
			if(tehai[i])
				syanten_kokusi--;
			if(tehai[i] >=2 && kokusi_toitu==0)
				kokusi_toitu=1;	
		}
	}             
	//字牌
	for(i=31;i<38;i++){      
		if(tehai[i]){	        
			syanten_kokusi--;
			if(tehai[i] >=2 && kokusi_toitu==0)
				kokusi_toitu=1;			            
		}
	}             
	//頭
	syanten_kokusi-= kokusi_toitu;             
	return syanten_kokusi;
}

//チートイシャンテン
int Syanten::TiitoituSyanten()
{
	int i=1,toitu=0,syurui=0,syanten_tiitoi;
	//トイツ数を数える
	for(;i<=37;i++){ 
		for(;!tehai[i];i++);
		if(i>=38) continue;
		syurui++;
		if(tehai[i] >=2)
		toitu++;
	}
	syanten_tiitoi=6-toitu;
	//４枚持ちを考慮
	if(syurui<7)
		syanten_tiitoi+= 7-syurui;
	return syanten_tiitoi;
}


//通常手シャンテン
int Syanten::NormalSyanten()
{
	//初期化
	mentu=0;
	toitu=0;
	kouho=0;
	temp=0;
	syanten_normal=8;
	for(int i=1;i<38;i++)
	{
		//頭抜き出し
		if(2 <= tehai[i])
		{            
			toitu++;
			tehai[i] -= 2;
			mentu_cut(1);
			tehai[i] += 2;
			toitu--;
		}
	}
    //フーロなしなら
	if(fuurosuu == 0)
        mentu_cut(1);   //頭無しと仮定して計算

	//死に孤立字牌(これバグるからだめ)
	bool ji=false;
	//for(int i=31;i<38;i++){if(tehai[i]==4)ji=true;}
	return syanten_normal-fuurosuu*2+ji;	//最終的な結果
}

//メンツ抜き出し
void Syanten::mentu_cut(int i)
{
	for(;!tehai[i];i++);
	if(i>=38){taatu_cut(1);return;}//メンツを抜き終わったのでターツ抜きへ
	//コーツ
	if(tehai[i]>=3)
	{
		mentu++;
		tehai[i]-=3;
		mentu_cut(i);
		tehai[i]+=3;
		mentu--;
	}
	//シュンツ
	if(tehai[i+1]&&tehai[i+2]&&i<30)
	{
		mentu++;
		tehai[i]--,tehai[i+1]--,tehai[i+2]--;
		mentu_cut(i);
		tehai[i]++,tehai[i+1]++,tehai[i+2]++;
		mentu--;
	}
	//メンツ無しと仮定
	mentu_cut(i+1);
}
//ターツ抜き出し
void Syanten::taatu_cut(int i)
{													   
	for(;!tehai[i];i++);
	if(i>=38) //抜き出し終了
	{
		temp=8-mentu*2-kouho-toitu;
		if(temp<syanten_normal) { syanten_normal=temp; }
		return;
	}
	if(mentu+kouho<4)
	{            
		//トイツ
		if(tehai[i]==2)
		{
			kouho++;
			tehai[i]-=2;
			taatu_cut(i);
			tehai[i]+=2;
			kouho--;
		}
	             
		//ペンチャンorリャンメン
		if(tehai[i+1]&&i<30)
		{
			kouho++;
			tehai[i]--,tehai[i+1]--;
			taatu_cut(i);
			tehai[i]++,tehai[i+1]++;
			kouho--;
		}
	             
		//カンチャン
		if(tehai[i+2]&&i<30&&i%10<=8)
		{
			kouho++;
			tehai[i]--,tehai[i+2]--;
			taatu_cut(i);
			tehai[i]++,tehai[i+2]++;
			kouho--;
		}
	}
	taatu_cut(i+1);
}