function tb = summarize_trajectories(r)
    n = numel(r.trajectories);
    tb = table('Size',[n 8], ...
        'VariableTypes', repmat("double",1,8), ...
        'VariableNames', {'k','row','ok_use','ok_mode','miss1','miss2','missT','n_steps'});

    for k = 1:n
        tr = r.trajectories(k);
        tb.k(k) = k;
        tb.row(k) = tr.idx;
        tb.ok_use(k) = double(tr.ok_use);
        if tr.ok_use && ~isempty(tr.stageB)
            if tr.stageB.ok; tb.ok_mode(k) = 1;      % ok
            else;            tb.ok_mode(k) = 2; end  % debug
            tb.miss1(k) = tr.stageB.miss(1);
            tb.miss2(k) = tr.stageB.miss(2);
            tb.miss3(k) = tr.stageB.miss(3); 
            tb.missT(k) = tr.stageB.miss(3);
        end
        if ~isempty(tr.transitions)
            tb.n_steps(k) = size(tr.transitions.state, 1);
        end
    end
end

tb2 = summarize_trajectories(results);
disp(tb2(tb2.ok_use==1, :))

fprintf('P2 miss1 中位数=%.1f, miss2 中位数=%.1f\n', ...
    median(tb2.miss1(tb2.ok_use==1)), median(tb2.miss2(tb2.ok_use==1)))