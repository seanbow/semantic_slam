% SCRIPT: define simulation constants
%
% $Id: DEFINE_SIMULATION.m 1061 2011-03-26 18:31:24Z joel $
% $LastChangedDate: 2011-03-26 13:31:24 -0500 (Sat, 26 Mar 2011) $
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

% time step size
DT=(1/100);
% Total number of propagation steps
% N = 47/DT; read from the trajectory config file
% degree to rad convertor
deg_to_rad = pi/180;

%% EARTH PARAMETERS
% G_VECTOR = [3.941862767012e-01;
%     6.971939837766e+00;
%     -6.907769460879e+00];

G_VECTOR = [0;0;-9.81];
        
% G_VECTOR = [0;0;0];

earth.g = G_VECTOR;

% lat=44.975301, long=-93.236163

%earth.omega = [0 ; 0 ; 7.292115e-5];
earth.omega = [0 ; 0 ; 0];

%local position in ECEF
earth.p = [-2.551357571614e+05;
         -4.512511736976e+06;
          4.485582568749e+06];

%% DATA SENSOR AVAILABILITY
% control inputs
% GPS_data_available=zeros(1,N);
% camera_data_available = zeros(1,N);
% encoder_velocity_data_available = zeros(1,N);
% 
% for j=2:N,
%     if mod(j,10000)==0
%         GPS_data_available(j)=0;
%     end
%     if mod(j,10)==2
%         camera_data_available(j)=0;
%     end
%     if mod(j,25)==0
%         encoder_velocity_data_available(j)=0;
%     end
% end


%% Noise Characteristics (Moving Robot)
% The following noise values are from the CB-IMU
% GYRO NOISES (Continuous time)
% sigma.r = 5.6e-5; % (rad/sec)/sqrt(Hz)(?) gyro measurement noise
% sigma.w = 1.6e-6; % (rad/sec^2)/sqrt(Hz)(?) gyro bias random walk noise

sigma.r = 0;
sigma.w = 0;

% ACCEL NOISES (Continuous time)
% sigma.g = 7.0e-4; % (m/sec^2)/sqrt(Hz) accel measurement noise
% sigma.v = 5.6e-5; % (m/sec^3)/sqrt(Hz) accel bias random walk noise

sigma.g = 0;
sigma.v = 0;

% FARAZ NOISE VALUES
%sigma.r = 2*5.6e-5;  % (rad/sec)/sqrt(Hz) gyro measurement noise - quantization step:4.1e-04;
%sigma.w = 1*1.6e-6;  % (rad/sec^2)/sqrt(Hz) gyro bias driving noise

% ACCEL NOISES (Continuous time)
%sigma.g = 2e-3;%7.0e-4;  % (m/sec^2)/sqrt(Hz) white noise component in accel meas/nt
%sigma.v = 5e-4;%5.6e-5;  % (m/sec^3)/sqrt(Hz) white noise driving bias rate in accels.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Testing
% GYRO NOISES (Continuous time)%
%sigma.r = 5.6e-3; % (rad/sec)/sqrt(Hz) rate random walk noise
%sigma.w = 1.6e-4; % (rad/sec^2)/sqrt(Hz) rate noise

% ACCEL NOISES (Continuous time)
%sigma.g = 7.0e-2; % (m/sec^2)/sqrt(Hz) white noise component in accel meas/nt
%sigma.v = 5.6e-3; % (m/sec^3)/sqrt(Hz) white noise driving bias rate in accels.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Motion Trajectory Parameters
% load waypoints
%  waypoints=load('joel.cfg','ascii');

% % my simple trajectory
N = 60; % how many seconds
theta = linspace(0, 2*pi * 180/pi * 5, N); % make 5 turns, express in degrees to stay consistent with earlier
phi = linspace(0, 2*pi * 180/pi, N);
psi = linspace(0, 2*pi * 180/pi, N);
waypoints = [[0:1:N-1]', 50 * cos( theta * pi/180 )', 50 * sin(theta * pi/180)', 50 * cos(theta * pi/180)', phi', psi', theta']';

trajt = waypoints(1,:);
pxatt = waypoints(2,:);
pyatt = waypoints(3,:);
pzatt = waypoints(4,:);
rollatt =  waypoints(5,:)*pi/180; %
pitchatt = waypoints(6,:)*pi/180; %
yawatt = waypoints(7,:)*pi/180;
% number of time steps
N = trajt(end)/DT;


%% Bias Initial Values

% gyro_bias_init = [-0.0031 -0.0034 -0.0058];
gyro_bias_init = [0;0;0];
  
  
% accel_bias_init = [0.0173 -0.1542 0.0310];
accel_bias_init = [0;0;0];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 

% % Laser to IMU transformation
% theta_x = 2.9546 * pi/180;
% theta_x = 0 * pi/180;
% theta_y = 23.24069 * pi/180;
% theta_y = 0 * pi/180;
% theta_z = 0.7 * pi/180;
% theta_z = 0 * pi/180;
% 
% s_R_lx = [1 0 0;
%           0 cos(theta_x) -sin(theta_x);
%           0 sin(theta_x) cos(theta_x)];
% s_R_ly = [cos(theta_y)  0 sin(theta_y);
%           0             1 0;
%           -sin(theta_y) 0 cos(theta_y)];
% s_R_lz = [cos(theta_z)  -sin(theta_z) 0;
%           sin(theta_z)  cos(theta_z) 0;
%           0 0 1];             
% %% Calibration of sick to isis
% i_P_laser = [0.255; -0.0135; 0.13];
% i_R_laser = [0 0 -1; 
%              0 1 0;
%              1 0 0] * s_R_lz * s_R_ly * s_R_lx;
% IMU2Las.q = rot2quat(i_R_laser);
% IMU2Las.p = i_P_laser;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 
%% Calibration of camera to imu
% i_P_camera = zeros(3,1);
i_P_camera = [.1;0;.1];

% i_R_camera = eye(3);
% i_R_camera = roty(15 * deg_to_rad) * rotz(-30 * deg_to_rad);
i_R_camera = rotz(pi/2);

IMU2Cam.q = rot2quat_eigen(i_R_camera);
IMU2Cam.p = i_P_camera;

init.q = [0;0;0;1];
% init.p = [7;4.5;1];
init.p = [0;0;0];

xtk = zeros(29,1);
%initial pose
xtk(1:4,1) = init.q;
xtk(14:16,1) = init.p;

% laser scanner noise std
sigma.rho = 0.01;
sigma.phi = 0.0001;

% camera noise
cam.c = [376 240];
cam.f = 800;
sigma.px = 1/cam.f;


% gps noise
sigma.gps = 1; % corresponding to 30 cm 1 sigma, 90 cm 3 sigma, etc.

%% Visual features;
nfeats = 1;
map.feats = zeros(3,nfeats);
map.feats(:,1:10) = randn(3,10);
map.nfeats = nfeats;
map.nfeats_valid = 0;
map.nfeats_per_im = 10;

%% Compass-Inclinometer Settings

sigma.att = 1*pi/180; % deg

