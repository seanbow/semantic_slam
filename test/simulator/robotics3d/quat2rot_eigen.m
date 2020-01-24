function R = quat2rot_eigen(q)

R = zeros(3,3);

tx  = 2*q(1);
ty  = 2*q(2);
tz  = 2*q(3);
twx = tx*q(4);
twy = ty*q(4);
twz = tz*q(4);
txx = tx*q(1);
txy = ty*q(1);
txz = tz*q(1);
tyy = ty*q(2);
tyz = tz*q(2);
tzz = tz*q(3);

R(1,1) = 1-(tyy+tzz);
R(1,2) = txy-twz;
R(1,3) = txz+twy;
R(2,1) = txy+twz;
R(2,2) = 1-(txx+tzz);
R(2,3) = tyz-twx;
R(3,1) = txz-twy;
R(3,2) = tyz+twx;
R(3,3) = 1-(txx+tyy);