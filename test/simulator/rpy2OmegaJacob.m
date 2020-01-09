function E = rpy2OmegaJacob(rpy)
% Jacobian of Omega in terms of rolldot, pitchdot and yawdot
%
% $Id: rpy2OmegaJacob.m 1044 2011-03-17 13:16:23Z joel $
% $LastChangedDate: 2011-03-17 08:16:23 -0500 (Thu, 17 Mar 2011) $
% Faraz Mirzaei
% faraz@cs.umn.edu

pitch = rpy(2);
yaw = rpy(3);

E = [ cos(yaw)*cos(pitch),           -sin(yaw),                   0;
      sin(yaw)*cos(pitch),            cos(yaw),                   0;
              -sin(pitch),                   0,                   1];