libname dat 'd:\work\dtOpt\data\';

proc sort data=dat.em_save_train;
	by _node_;
run;

proc freq data=dat.accepts;
	table bad;
run;

proc univariate data=dat.accepts;
	var bureau_score ltv tot_rev_line;
	histogram;
run;

data dtnodes;
	set dat.em_save_train;
	by _node_;
	retain n_good n_bad bad_amt;
	if first._node_ then do;
		n_good = 1 - bad;
		n_bad = bad;
		bad_amt = loan_amt * bad;
	end;
	else do;
		n_good + 1 - bad;
		n_bad + bad;
		bad_amt + loan_amt * bad;
	end;
	if last._node_ then output;
	keep _node_ n_good n_bad bad_amt;
run;

/*proc export data=dtnodes */
/*	outfile='d:\data\dtnodes.csv'*/
/*	dbms=csv replace;*/
/*run;*/

data bad_rate;
	set dtnodes;
	n_samp = n_good + n_bad;
	pct_bad = round(n_bad / n_samp * 100, 0.01);
run;

proc print data=bad_rate;
run;

proc optmodel;
	set<num> node;
	num n_good{node}, n_bad{node}, bad_amt{node};
	read data dtnodes into node=[_node_] n_good n_bad bad_amt; /*读入数据*/
	var w{node} binary; /*变量，binary指定取值范围是0/1*/
	impvar tot_bad = sum{i in node}(w[i]*n_bad[i]);
	impvar sum_bad_amt = sum{i in node}(w[i]*bad_amt[i]);
	max tot_samp = sum{i in node}(w[i]*(n_good[i]+n_bad[i])); /*优化目标*/
	impvar bad_rate = tot_bad / tot_samp;
	con c1: tot_bad <= tot_samp * 0.1; /*比率约束*/
	con c2: sum_bad_amt <= 5e6; /*金额约束*/
	solve with milp; /*求解器：混合整数线性规划*/
	print w bad_rate sum_bad_amt tot_samp;
run;