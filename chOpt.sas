proc import datafile='d:\data\ch_opt.csv' 
	dbms=csv out=chinfo replace;
run;

%let PLATFORM_FEE_RATE = 0.3;

data chinfo2;
	set chinfo;
	'ģ��ͨ����'n = '�ͻ���'n / '������'n;
	'�����Ž��'n = '�ͻ���'n * 'ƽ�����Ŷ��'n;
	'�ſ���'n = '�����Ž��'n * '֧����'n;
	'��Ϣ����'n = '�ſ���'n * '��Ȩ����'n * (1 - &platform_fee_rate);
	'��ʧ���'n = '�ſ���'n * '��ʧ��'n;
	'FTP�۳�ǰ����'n = '��Ϣ����'n - '��ʧ���'n;
	'FTP�۳�ǰ������'n = 'FTP�۳�ǰ����'n / '�ſ���'n;

	keep 'ģ��ͨ����'n '��ʧ��'n '��Ȩ����'n 'FTP�۳�ǰ����'n;
	rename 'ģ��ͨ����'n = r_pass
		'��ʧ��'n = r_loss
		'��Ȩ����'n = interest
		'FTP�۳�ǰ����'n = profit;
run;

%let MIN_APPROVAL_RATE = 0.72;
%let MAX_LOSS_RATE = 0.025;
%let MIN_WEIGHTED_INTEREST_RATE = 0.13;

proc optmodel;
	set<num> n;
	num r_pass{n}, r_loss{n}, interest{n}, profit{n};
	read data chinfo2 into n=[_n_] r_pass r_loss interest profit;
	var p{n} >=0 <=1;
	max tot_profit = sum{i in n}(p[i] * profit[i]);
	con c0: sum{i in n}p[i]=1;
	con c1: p[2] >= 0.2;
	con c2: p[3] >= 0.2;
	con c3: p[4] >= 0.35;
	impvar w_pass = sum{i in n}(p[i] * r_pass[i]);
	impvar w_loss = sum{i in n}(p[i] * r_loss[i]);
	impvar w_interest = sum{i in n}(p[i] * interest[i]);
	con c4: w_pass >= &min_approval_rate;
	con c5: w_loss <= &max_loss_rate;
	con c6: w_interest >= &min_weighted_interest_rate;
	solve with lp;
	print p tot_profit w_loss w_pass w_interest;
run;
