function validate_dynamics_sym()
%VALIDATE_DYNAMICS_SYM 符号动力学 casadi_dynamics 与数值重构的对拍。
%   验证 18 维 Xdot 在多组交战几何下数值版与符号版一致 (规避段, 不含饱和)。

    setup_expert_path();
    import casadi.*
    H = evasion_helpers();

    w = 1e4; KD1 = 4; KD2 = 4; r_star = 20; t = 0.5;
    gamma = 0.1; wT = 1e3; wE = 1.44e3;
    theta = [gamma; wT; wE];
    params = struct('w', w, 'KD1', KD1, 'KD2', KD2, 'r_star', r_star);
    a1 = w / (1 + exp(-gamma));
    a2 = w * exp(-gamma) / (1 + exp(-gamma));

    % 符号 Function
    Xs = SX.sym('X', 18); ths = SX.sym('th', 3); ts = SX.sym('t');
    Xdot_s = casadi_dynamics(Xs, ths, ts, params);
    fdyn = Function('fdyn', {Xs, ths, ts}, {Xdot_s});

    cases = {
        [10000,6000,3500,deg2rad(-20),deg2rad(155),500], [1000,0,0,deg2rad(40),deg2rad(-20),600], [-1000,0,0,deg2rad(40),deg2rad(-20),600]
        [ 7000,3000,1500,deg2rad(-10),deg2rad(170),500], [ 800,0,0,deg2rad(30),deg2rad(-10),600], [ -600,200,0,deg2rad(50),deg2rad(-25),600]
        [ 8500,4500,-2000,deg2rad(15),deg2rad(160),500], [ 500,100,0,deg2rad(35),deg2rad(  5),600], [ -900,-300,0,deg2rad(45),deg2rad(-15),600]
    };
    Tstate = [0,0,0,0,0,0];

    fprintf('\n=== 符号动力学 vs 数值 对拍 (gamma=%.2f wT=%.0f wE=%.0f) ===\n', gamma, wT, wE);
    maxrel = 0;
    for k = 1:size(cases,1)
        M = cases{k,1}; D1 = cases{k,2}; D2 = cases{k,3};
        X = [M(:); D1(:); D2(:)];

        % --- 数值 Xdot ---
        [q_y_MD1,q_z_MD1,r_MD1] = H.computeLOS(D1(1:3), M(1:3));
        [q_y_MD2,q_z_MD2,r_MD2] = H.computeLOS(D2(1:3), M(1:3));
        [q_y_TM,q_z_TM,r_TM]   = H.computeLOS(Tstate(1:3), M(1:3));
        [r_dot_MD1,qd_y1,qd_z1] = H.computeMDDot(M, D1, r_MD1, q_y_MD1, q_z_MD1);
        [r_dot_MD2,qd_y2,qd_z2] = H.computeMDDot(M, D2, r_MD2, q_y_MD2, q_z_MD2);
        [r_dot_TM,qd_y_TM,qd_z_TM] = H.computeMDDot(M, Tstate, r_TM, q_y_TM, q_z_TM);
        tf1 = t - r_MD1/r_dot_MD1; tf2 = t - r_MD2/r_dot_MD2; tfT = t - r_TM/r_dot_TM;
        aD1 = H.computePN(KD1, D1(6), qd_y1, qd_z1, D1(4));
        aD2 = H.computePN(KD2, D2(6), qd_y2, qd_z2, D2(4));
        aM = H.computeOptimalCmd_duo_target( ...
            q_y_MD1,q_z_MD1,qd_y1,qd_z1,r_MD1,r_dot_MD1,t,tf1, ...
            q_y_MD2,q_z_MD2,qd_y2,qd_z2,r_MD2,r_dot_MD2,tf2, ...
            q_y_TM,q_z_TM,qd_y_TM,qd_z_TM,r_TM,r_dot_TM,tfT, ...
            M(4),M(5),D1(4),D1(5),D2(4),D2(5), aD1,aD2,a1,a2,wT,wE,r_star);
        Xdot_num = [kin(M,aM); kin(D1,aD1); kin(D2,aD2)];

        % --- 符号 Xdot ---
        Xdot_sym = full(fdyn(X, theta, t));

        rel = norm(Xdot_num - Xdot_sym) / max(norm(Xdot_num), 1e-6);
        maxrel = max(maxrel, rel);
        fprintf('case%d: 相对误差=%.3e  (|Xdot|=%.3f)\n', k, rel, norm(Xdot_num));
    end
    fprintf('--- 最大相对误差 = %.3e %s ---\n', maxrel, ...
            ternary(maxrel < 1e-6, '(对拍通过)', '(需检查)'));
end

function d = kin(s, a)
    theta = s(4); psi_v = s(5); V = s(6);
    d = [V*cos(theta)*cos(psi_v); V*sin(theta); -V*cos(theta)*sin(psi_v);
         a(1)/V; -a(2)/(V*cos(theta)); 0];
end

function s = ternary(c, a, b)
    if c; s = a; else; s = b; end
end
