proc import datafile='d:\work\dtOpt\data\chopt.csv' 
	dbms=csv out=chinfo replace;
run;

%let PLATFORM_FEE_RATE = 0.3;

data chinfo2;
	set chinfo;
	'模型通过率'n = '客户数'n / '进件数'n;
	'总授信金额'n = '客户数'n * '平均授信额度'n;
	'放款金额'n = '总授信金额'n * '支用率'n;
	'利息收入'n = '放款金额'n * '加权利率'n * (1 - &platform_fee_rate);
	'损失金额'n = '放款金额'n * '损失率'n;
	'FTP扣除前利润'n = '利息收入'n - '损失金额'n;
	'FTP扣除前利润率'n = 'FTP扣除前利润'n / '放款金额'n;

	keep '模型通过率'n '损失率'n '加权利率'n 'FTP扣除前利润'n;
	rename '模型通过率'n = r_pass
		'损失率'n = r_loss
		'加权利率'n = interest
		'FTP扣除前利润'n = profit;
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
