function ident = stage2_ident_commit(ident, a_M_sat)
%STAGE2_IDENT_COMMIT  提交本拍实际饱和加速度，供下一拍因果 RLS。
    if isempty(ident)
        return;
    end
    ident.a_M_sat_prev = double(a_M_sat(:));
    ident.has_prev = true;
end
