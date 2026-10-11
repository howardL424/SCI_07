function minima = ps_reference_minima(x0, Tev)
%ps_reference_minima - Audit the fixed-weight main route on the same window
%   MINIMA = ps_reference_minima(X0,TEV) returns post-step defender minima.
%   See also test_ps_core, stage2_step

arguments
    x0 (18,1) double {mustBeFinite}
    Tev (1,1) double {mustBePositive,mustBeFinite}
end
core=stage2_core_init(struct('dt',0.1));
% Comparison only: the shared warm weights, encoded in the native action.
% No reference commands or weights enter the direct expert/policy path.
a=[log10(1e4/1440)-1;(log10(1e3/1440)-0.5)/2.5;0];
s=x0; t=0; minima=[inf;inf];
while t<Tev
    [~,s,t,fact]=stage2_step(core,s,t,Tev,a);
    minima=min(minima,[fact.r_MD1;fact.r_MD2]);
    if fact.done; break; end
end
end
