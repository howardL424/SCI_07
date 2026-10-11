function report = test_ps_core()
%test_ps_core - Verify ballistic propagation, equivalent input and rewards
%   REPORT = test_ps_core() runs physical consistency checks.
%   See also ps_step, stage2_step

core=ps_core_init();
s=[-10000;500;0;0.1;0.2;500;1000;1000;1000;0;0;600; ...
    1200;1500;1000;0;0;600];
[~,sn]=ps_step(core,s,0,10,[0;0]);
v=500*[cos(.1)*cos(.2);sin(.1);-cos(.1)*sin(.2)];
ballistic=max(abs(sn(1:3)-(s(1:3)+.1*v)));
assert(ballistic<1e-6);
old=stage2_core_init(struct('dt',.1));
[~,old_sn,~,fact]=stage2_step(old,s,0,10,[0;0;0]);
[~,new_sn]=ps_step(core,s,0,10,fact.a_M_sat/core.max_ov_M);
equivalent=max(abs(old_sn-new_sn)); assert(equivalent<1e-9);
rewards=zeros(3,1);
for k=1:3
    f=struct('r_MD1',10*k,'r_MD2',12*k,'intercepted',k==3);
    rewards(k)=stage2_process_reward(core,f);
    assert(rewards(k)==stage2_process_reward(old,f));
end
report=struct('ballistic_error',ballistic,'equivalent_error',equivalent, ...
    'process_rewards',rewards,'meta',ps_transition_meta()); disp(report);
end
