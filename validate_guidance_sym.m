function validate_guidance_sym()
%VALIDATE_GUIDANCE_SYM 数值版(式47) computeOptimalCmd_duo_target 与
%   CasADi 符号版 casadi_guidance_duo_target 的逐点对拍。
%   在若干真实交战几何下分别求 uA, 比较相对误差。

    import casadi.*
    H = evasion_helpers();

    % ---- 构造符号 Function (37 个标量入参) ----
    in = SX.sym('in', 37);
    args = arrayfun(@(k) in(k), 1:37, 'UniformOutput', false);
    uA_sym = casadi_guidance_duo_target(args{:});
    fsym = Function('fsym', {in}, {uA_sym});

    % ---- 测试用交战几何 (M, D1, D2; 目标在原点) ----
    cases = {
        [10000,6000,3500,deg2rad(-20),deg2rad(155),500], [1000,0,0,deg2rad(40),deg2rad(-20),600], [-1000,0,0,deg2rad(40),deg2rad(-20),600]
        [ 7000,3000,1500,deg2rad(-10),deg2rad(170),500], [ 800,0,0,deg2rad(30),deg2rad(-10),600], [ -600,200,0,deg2rad(50),deg2rad(-25),600]
        [ 8500,4500,-2000,deg2rad(15),deg2rad(160),500], [ 500,100,0,deg2rad(35),deg2rad(  5),600], [ -900,-300,0,deg2rad(45),deg2rad(-15),600]
        [ 5000,2000,800,deg2rad(-5),deg2rad(175),500], [ 300,50,0,deg2rad(25),deg2rad(-8),600], [ -400,80,0,deg2rad(40),deg2rad(-30),600]
    };
    T = [0,0,0]; T_state = [0,0,0,0,0,0];
    w1 = 1e4; w2 = 1e4; wT = 1e3; wE = 1.44e3; r_star = 20; t = 0.5;

    fprintf('\n=== 式47 数值 vs 符号 对拍 (t=%.2f, wT=%.0f, wE=%.0f) ===\n', t, wT, wE);
    maxrel = 0;
    for k = 1:size(cases,1)
        M = cases{k,1}; D1 = cases{k,2}; D2 = cases{k,3};

        % --- 几何 (与 simulate_warmstart 一致) ---
        [q_theta,q_psi,r_MT]   = H.computeLOS(M(1:3), T);
        [q_y_MD1,q_z_MD1,r_MD1] = H.computeLOS(D1(1:3), M(1:3));
        [q_y_MD2,q_z_MD2,r_MD2] = H.computeLOS(D2(1:3), M(1:3));
        [q_y_TM,q_z_TM,r_TM]   = H.computeLOS(T_state(1:3), M(1:3));

        [r_dot_MD1,qd_y1,qd_z1] = H.computeMDDot(M, D1, r_MD1, q_y_MD1, q_z_MD1);
        [r_dot_MD2,qd_y2,qd_z2] = H.computeMDDot(M, D2, r_MD2, q_y_MD2, q_z_MD2);
        [r_dot_TM,qd_y_TM,qd_z_TM] = H.computeMDDot(M, T_state, r_TM, q_y_TM, q_z_TM);

        rdot1 = sign(r_dot_MD1)*max(abs(r_dot_MD1),1e-3); tf1 = t - r_MD1/rdot1;
        rdot2 = sign(r_dot_MD2)*max(abs(r_dot_MD2),1e-3); tf2 = t - r_MD2/rdot2;
        rdotT = sign(r_dot_TM )*max(abs(r_dot_TM ),1e-3); tfT = t - r_TM /rdotT;

        % 防御弹 PN 过载
        aD1 = H.computePN(4, D1(6), qd_y1, qd_z1, D1(4));
        aD2 = H.computePN(4, D2(6), qd_y2, qd_z2, D2(4));

        % --- 数值版 ---
        uA_num = H.computeOptimalCmd_duo_target( ...
            q_y_MD1, q_z_MD1, qd_y1, qd_z1, r_MD1, r_dot_MD1, t, tf1, ...
            q_y_MD2, q_z_MD2, qd_y2, qd_z2, r_MD2, r_dot_MD2, tf2, ...
            q_y_TM, q_z_TM, qd_y_TM, qd_z_TM, r_TM, r_dot_TM, tfT, ...
            M(4), M(5), D1(4), D1(5), D2(4), D2(5), ...
            aD1, aD2, w1, w2, wT, wE, r_star);

        % --- 符号版 ---
        invec = [q_y_MD1; q_z_MD1; qd_y1; qd_z1; r_MD1; r_dot_MD1; t; tf1; ...
                 q_y_MD2; q_z_MD2; qd_y2; qd_z2; r_MD2; r_dot_MD2; tf2; ...
                 q_y_TM; q_z_TM; qd_y_TM; qd_z_TM; r_TM; r_dot_TM; tfT; ...
                 M(4); M(5); D1(4); D1(5); D2(4); D2(5); ...
                 aD1(1); aD1(2); aD2(1); aD2(2); w1; w2; wT; wE; r_star];
        uA_s = full(fsym(invec));

        rel = norm(uA_num - uA_s) / max(norm(uA_num), 1e-6);
        maxrel = max(maxrel, rel);
        fprintf(['case%d: num=[% .4f % .4f]  sym=[% .4f % .4f]  ', ...
                 '相对误差=%.2e\n'], k, uA_num(1), uA_num(2), uA_s(1), uA_s(2), rel);
    end
    fprintf('--- 最大相对误差 = %.3e %s ---\n', maxrel, ...
            ternary(maxrel < 1e-6, '(对拍通过)', '(需检查)'));
end

function s = ternary(c, a, b)
    if c; s = a; else; s = b; end
end
