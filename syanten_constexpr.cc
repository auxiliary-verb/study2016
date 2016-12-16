
constexpr

int Syanten::KokusiSyanten()
{
	int kokusi_toitu=0,syanten_kokusi=13,i;             
	for(i=1;i<30;i++){        
		if(i%10==1||i%10==9||i%20==1||i%20==9){	   
			if(tehai[i])
				syanten_kokusi--;
			if(tehai[i] >=2 && kokusi_toitu==0)
				kokusi_toitu=1;	
		}
	}             
	for(i=31;i<38;i++){      
		if(tehai[i]){	        
			syanten_kokusi--;
			if(tehai[i] >=2 && kokusi_toitu==0)
				kokusi_toitu=1;			            
		}
	}             
	syanten_kokusi-= kokusi_toitu;             
	return syanten_kokusi;
}

int Syanten::TiitoituSyanten()
{
	int i=1,toitu=0,syurui=0,syanten_tiitoi;
	for(;i<=37;i++){ 
		for(;!tehai[i];i++);
		if(i>=38) continue;
		syurui++;
		if(tehai[i] >=2)
		toitu++;
	}
	syanten_tiitoi=6-toitu;
	if(syurui<7)
		syanten_tiitoi+= 7-syurui;
	return syanten_tiitoi;
}

int Syanten::NormalSyanten()
{
	int result=99;
	int tmpresult=0;
	for(int i=1;i<38;i++)
	{
		if(2 <= tehai[i])
		{            
			tehai[i] -= 2;
			tmpresult=checkNormalSyanten()-1;
			if(tmpresult < result){
				result=tmpresult;
			}
			tehai[i] += 2;
		}
	}
	
	if(fuurosuu == 0){
        tmpresult=checkNormalSyanten();
		if(tmpresult < result){
			result=tmpresult;
		}
	}


	return result;
}
