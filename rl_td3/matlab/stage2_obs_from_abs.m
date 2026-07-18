function obs = stage2_obs_from_abs(core, state_abs, t, Tev)
%STAGE2_OBS_FROM_ABS  由绝对态构造 13 维相对观测（不推进动力学）。

    H = core.H;
    M  = state_abs(1:6)';
    D1 = state_abs(7:12)';
    D2 = state_abs(13:18)';
    T_pos   = core.T_pos;
    T_state = core.T_state; %#ok<NASGU>

    [qy_D1M, qz_D1M, r_MD1] = H.computeLOS(D1(1:3), M(1:3));
    [qy_D2M, qz_D2M, r_MD2] = H.computeLOS(D2(1:3), M(1:3));
    [qy_MT,  qz_MT,  r_MT]  = H.computeLOS(M(1:3),  T_pos);
    [rdot_MD1, ~, ~] = H.computeMDDot(M, D1, r_MD1, qy_D1M, qz_D1M);
    [rdot_MD2, ~, ~] = H.computeMDDot(M, D2, r_MD2, qy_D2M, qz_D2M);

    tgo_norm = max(0.0, (Tev - t) / max(Tev, 1e-6));
    obs = [r_MD1; rdot_MD1; qy_D1M; qz_D1M;
           r_MD2; rdot_MD2; qy_D2M; qz_D2M;
           r_MT;  qy_MT;   qz_MT;  M(6);  tgo_norm];
end
