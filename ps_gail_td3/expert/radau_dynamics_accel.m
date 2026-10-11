function [dx, aM, aD1, aD2] = radau_dynamics_accel(x, u, params)
%radau_dynamics_accel - Evaluate symbolic direct control and defender PN
%   [DX,AM,AD1,AD2] = radau_dynamics_accel(X,U,PARAMS) evaluates Eq.5/9.
%   See also build_radau_nlp_accel, ps_step

arguments
    x
    u
    params (1,1) struct
end
M=x(1:6); D1=x(7:12); D2=x(13:18);
% U is physical acceleration, constrained to the 8g ball by the NLP.
aM=u;
aD1=defender(M,D1,params.KD1,params.amax_D);
aD2=defender(M,D2,params.KD2,params.amax_D);
dx=[kin(M,aM);kin(D1,aD1);kin(D2,aD2)];
end

function a=defender(M,D,K,amax)
p=M(1:3)-D(1:3); r=sqrt(p.'*p+1e-6);
qy=asin(p(2)/r); qz=-atan2(p(3),p(1));
dy=(D(6)*(cos(D(4))*sin(qy)*cos(qz-D(5))-sin(D(4))*cos(qy)) ...
    -M(6)*(cos(M(4))*sin(qy)*cos(qz-M(5))-sin(M(4))*cos(qy)))/r;
dz=(D(6)*cos(D(4))*sin(qz-D(5)) ...
    -M(6)*cos(M(4))*sin(qz-M(5)))/(r*sqrt(cos(qy)^2+1e-10));
a=[K*D(6)*dy;-K*D(6)*dz*cos(D(4))];
s=sqrt(a.'*a+1e-9);
smax=0.5*(s+amax+sqrt((s-amax)^2+(0.05*amax)^2));
a=a*(amax/smax);
end

function d=kin(s,a)
d=[s(6)*cos(s(4))*cos(s(5));s(6)*sin(s(4)); ...
    -s(6)*cos(s(4))*sin(s(5));a(1)/s(6); ...
    -a(2)/(s(6)*cos(s(4)));0];
end
