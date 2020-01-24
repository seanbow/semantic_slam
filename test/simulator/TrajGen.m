function [ztk_1,xtk_1]=TrajGen(step, xtk, DT, sigma, traj, earth)

% Trajectory Generation Generates the state vector for KF15 at timestep 
% 'step' using the polynomial coefficients in traj. These noise-free values
% are returned in xtk_1. The function also computes noisy and biased
% measuremnts of the IMU (i.e., rotational velocity and linear
% acceleration) using noise values in sigma, previous bias in xtk, and
% earth gravity and rotation parameters in earth.
%
% $Id: TrajGen.m 1044 2011-03-17 13:16:23Z joel $
% $LastChangedDate: 2011-03-17 08:16:23 -0500 (Thu, 17 Mar 2011) $
%-------------------------------------------------------------------
% xtk - REAL DATA
%
% LOCAL to GLOBAL - ORIENTATION
% 1 q1
% 2 q2
% 3 q3
% 4 q4
%
% SENSOR BIAS - GYRO
% 5 gyro_bias a
% 6 gyro_bias b
% 7 gyro_bias c
%
% wrt GLOBAL - VELOCITY
% 8 v1
% 9 v2
% 10 v3
%
% SENSOR BIAS - ACCEL
% 11 accel_bias a
% 12 accel_bias b
% 13 accel_bias c
%
% wrt GLOBAL - POSITION
% 14 r1
% 15 r2
% 16 r3
%
% LOCAL (REAL) DRIVING ROTATIONAL VELOCITY
% 17 omega a
% 18 omega b
% 19 omega c
%
% LOCAL (REAL) DRIVING ACCELERATION
% 20 dot_v a
% 21 dot_v b
% 22 dot_v c
%
%
%-----------------------------------------------------------------------
% ztk
%
% LOCAL ROTATIONAL VELOCITY DRIVING MEAS/RED by gyros (IMU rate)
% 1 gyro a
% 2 gyro b
% 3 gyro c
%
% LOCAL DRIVING ACCELERATIONS MEAS/RED by accelerometers (IMU rate)
% 4 accel a
% 5 accel b
% 6 accel c
%


%% INITIALIZATIONS

time = step*DT;

% real state vector 
xtk_1 = zeros(22,1);


%% ORIENTATION & OMEGA COMPUTATION

roll = ppval(time, traj.roll);
pitch = ppval(time, traj.pitch);
yaw = ppval(time, traj.yaw);
rpy = [roll; pitch; yaw];

rot = rotz(yaw)*roty(pitch)*rotx(roll);
% rot = rotx(roll) * roty(pitch) * rotz(yaw);
g_R_i = rot;

% the quaternion in this time step
xtk_1(1:4,1) = rot2quat_eigen(g_R_i);
qk_1_r = xtk_1(1:4,1);

rolldot = ppval(time, traj.rolldot);
pitchdot = ppval(time, traj.pitchdot);
yawdot = ppval(time, traj.yawdot);

% rotational velocity in IMU frame
xtk_1(17:19,1) = g_R_i'*rpy2OmegaJacob(rpy)*[rolldot; pitchdot; yawdot];


%% GYRO BIASES PROPAGATION

% additive noise components
gyro_noise_bias = sqrt(DT)*sigma.w*randn(3,1);
% real gyro bias propagation
xtk_1(5:7) = xtk(5:7) + gyro_noise_bias ;



%% VELOCITY PROPAGATION

% position in global frame
posx = ppval(time, traj.x);
posy = ppval(time, traj.y);
posz = ppval(time, traj.z);
xtk_1(14:16,1) = [posx; posy; posz];

velx = ppval(time, traj.velx);
vely = ppval(time, traj.vely);
velz = ppval(time, traj.velz);
xtk_1(8:10,1) = [velx; vely; velz];

accelx = ppval(time, traj.accelx);
accely = ppval(time, traj.accely);
accelz = ppval(time, traj.accelz);
g_a_real = [accelx; accely; accelz];
i_a_real = g_R_i'*g_a_real;

% local acceleration
xtk_1(20:22) = i_a_real(1:3);



%% ACCELEROMETER BIASES PROPAGATION

% Bias/real(k+1) = Bias/real(k) + noise x DT
% additive noise components
accel_noise_bias = sqrt(DT) * sigma.v * randn(3,1) ;
% real accel bias propagation
xtk_1(11:13) = xtk(11:13) + accel_noise_bias ;


%% GYRO SENSOR MEASUREMENTS

% gyro noise white component
gyro_noise = 1/sqrt(DT)*sigma.r*randn(3,1);

% gyro meas/nt = real rotational velocity + real bias + gyro noise
ztk_1(1:3,1) = xtk_1(17:19) + xtk_1(5:7) + gyro_noise + quat2rot_eigen(qk_1_r)*earth.omega;


%% ACCELEROMETER SENSOR  MEASUREMENTS

% These are the measurements recorded by the accelerometers
% accelerometer noise white component
accel_noise = sqrt(DT) * sigma.g * randn(3,1);

% accel meas/nt = real accel (local) + real accel bias
%                 + accel white noise - gravitational acceleration (local)
ztk_1(4:6,1) = xtk_1(20:22,1) + xtk_1(11:13,1) + accel_noise ...
               -quat2rot_eigen(qk_1_r)'*earth.g...
               +quat2rot_eigen(qk_1_r)'*skewsymm(earth.omega)...
               *(2*xtk_1(8:10)+skewsymm(earth.omega)*earth.p);

           
%% Compass-Inclinometer Measurements
dtheta = randn(3,1)*sigma.att;
dq = [dtheta/2 ; 1];

ztk_1(7:10,:) = quat_mul_eigen(dq,qk_1_r);

%% GPS Measurements (to be added)
