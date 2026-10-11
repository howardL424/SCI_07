function ps_expert_setup()
%ps_expert_setup - Locate branch helpers and the installed CasADi package
%   ps_expert_setup() adds read-only dependencies and locates CasADi.
%   See also build_radau_nlp_accel

branch=fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(branch,'matlab'));
ps_core_init();
if isempty(which('casadi.Opti'))
    candidates={fullfile(matlabroot,'toolbox','casadi'), ...
        fullfile('E:','matlab','casadi-3.7.2-windows64-matlab2018b')};
    for k=1:numel(candidates)
        if isfile(fullfile(candidates{k},'casadiMEX.mexw64'))
            addpath(candidates{k}); break
        end
    end
end
assert(~isempty(which('casadi.Opti')),'CasADi is not installed');
end
