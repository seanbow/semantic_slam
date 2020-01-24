%
% Real World Simulator Script
%
% $Id: SCRIPT.m 1061 2011-03-26 18:31:24Z joel $
% $LastChangedDate: 2011-03-26 13:31:24 -0500 (Sat, 26 Mar 2011) $
% Faraz Mirzaei
% faraz@cs.umn.edu
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

clc; clear; clear global; close all;


addpath robotics3d

fprintf('Prepare the data directory ...\n');
% prepare the directory
current_path = pwd;
cd data
!rm *.*
cd(current_path)

USE_MAT = 1;

%% INITIALIZATIONS

% generate a given scenario
% load rn_seq
% randn('state',rn_seq1); rand('state',rn_seq2);

% generate a new random scenario and save the seeds
rn_seq1 = randn('state');
rn_seq2 = rand('state');
save rn_seq;

% define the constants
DEFINE_SIMULATION

% real state vector
xtk = zeros(22,1);
%initial bias
xtk(5:7,1) = zeros(3,1);gyro_bias_init;
xtk(11:13) = zeros(3,1);accel_bias_init;

xtk_log = [];

%% Generate Spline Polynomials and derivatives
traj.t = trajt;
% get the static intervals as well as the polynomials
[traj.x, static_intervals] = PoseGen1D(trajt,pxatt);
traj.y = PoseGen1D(trajt,pyatt);
traj.z = PoseGen1D(trajt,pzatt);
traj.velx = fnder(traj.x);
traj.vely = fnder(traj.y);
traj.velz = fnder(traj.z);
traj.accelx = fnder(traj.x,2);
traj.accely = fnder(traj.y,2);
traj.accelz = fnder(traj.z,2);

traj.roll = PoseGen1D(trajt,rollatt);
traj.pitch = PoseGen1D(trajt,pitchatt);
traj.yaw = PoseGen1D(trajt,yawatt);
traj.rolldot = fnder(traj.roll);
traj.pitchdot = fnder(traj.pitch);
traj.yawdot = fnder(traj.yaw);


%% the laser scan storage
Scanr = zeros(floor(N/10) + 1, 182);
Scanm = zeros(floor(N/10) + 1, 182);
Scanid = zeros(floor(N/10) + 1, 182);
LasInd = 1;

%% image storage
ImgPtsU = zeros(floor(N/30), map.nfeats_per_im);
ImgPtsV = zeros(floor(N/30), map.nfeats_per_im);
ImgPtsUm = zeros(floor(N/30), map.nfeats_per_im);
ImgPtsVm = zeros(floor(N/30), map.nfeats_per_im);
ImgId = zeros(floor(N/30), map.nfeats_per_im);
ImgTS = zeros(floor(N/30),1);
ImgInd = 1;

%% state vector storage
Qr = zeros(4, N+1);
GBr = zeros(3,N+1);
Vr = zeros(3,N+1);
ABr = zeros(3,N+1);
Pr = zeros(3,N+1);
Or = zeros(3,N+1);
Ar = zeros(3,N+1);
Om = zeros(3,N+1);
Am = zeros(3,N+1);
Qm = zeros(4,N+1);

imu_times = zeros(N+1,1);

%% storage for GPS
GPS = zeros(3, floor(N/100) + 1);
GPSm = zeros(3, floor(N/100) + 1);
GPSTS = zeros(1,floor(N/100) + 1);
GPSInd = 1;

%% Main Loop: Generate the trajectory
fprintf('Generating the trajectory and measurements ...\n');

for i=0:N,
  if mod(i,100)==0
    fprintf('Time Step: %i (%2.0f%%) \n',i,i/N*100);
  end
  
  [ztk_1,xtk_1]=TrajGen(i, xtk, DT, sigma, traj, earth);
  
  xtk_log(:,i+1) = xtk_1;
  
  Qr(:,i+1) = xtk_1(1:4);
  GBr(:,i+1) = xtk_1(5:7);
  Vr(:,i+1) = xtk_1(8:10);
  ABr(:,i+1) = xtk_1(11:13);
  Pr(:,i+1) = xtk_1(14:16);
  Or(:,i+1) = xtk_1(17:19);
  Ar(:,i+1) = xtk_1(20:22);
  
  Om(:,i+1) = ztk_1(1:3);
  Am(:,i+1) = ztk_1(4:6);
  Qm(:,i+1) = ztk_1(7:10);
  
%   imu_times(i+1) = 10*(i+1);
  imu_times(i+1) = i * DT;
  
  xtk=xtk_1;
  
  % Image recorded every 30 IMU cycles, corresponding to ~3Hz
  if mod(i+1,30)==0
    [img, map] = CameraObs(xtk, sigma, map, IMU2Cam);
    
    ImgPtsU(ImgInd,:) = img.feats(1,:);
    ImgPtsV(ImgInd,:) = img.feats(2,:);
    ImgPtsUm(ImgInd,:) = img.feats_meas(1,:);
    ImgPtsVm(ImgInd,:) = img.feats_meas(2,:);
    
    ImgId(ImgInd,:) = img.feat_id;
    ImgTS(ImgInd) = 10*(i+1);
    
    ImgInd = ImgInd + 1;
  end
  
  % GPS recorded every 100 IMU cycles, correpsonding to 1Hz
  if mod(i+1, 50) == 0
    GPS(:, GPSInd) = Pr(:,i+1);
    GPSm(:, GPSInd) = Pr(:,i+1) + sigma.gps * randn(3,1);
    GPSTS(GPSInd) = 10 * (i+1);
    GPSInd = GPSInd + 1;
  end
end

%% Write the laser data files
fprintf('Writing data files ...\n');

if USE_MAT
  save data/cam.mat ImgPtsU ImgPtsV ImgTS ImgPtsUm ImgPtsVm ImgId;
  save data/imu.mat Or Ar Om Am Qr Pr Qm imu_times;
  save data/state.mat xtk_log;
  save data/map.mat map;
  save data/IMU2Cam.mat IMU2Cam;
  save data/earth.mat earth;
  save data/sigma.mat sigma;
  save data/gps.mat GPS GPSm GPSTS;
else
  
  fid1 = fopen('data/camera_data_true.dat','w+');
  fid2 = fopen('data/camera_data_meas.dat','w+');
  fid3 = fopen('data/camera_data_id.dat','w+');
  for i=1:size(ImgPtsU,1)
    fprintf(fid1, '%s ', num2str(ImgPtsU(i,:), '%.14f '));
    fprintf(fid1, '%s ', num2str(ImgPtsV(i,:), '%.14f '));
    fprintf(fid1, '%s\n', num2str(ImgTS(i,:), '%g '));
    
    fprintf(fid2, '%s ', num2str(ImgPtsUm(i,:), '%.14f '));
    fprintf(fid2, '%s ', num2str(ImgPtsVm(i,:), '%.14f '));
    fprintf(fid2, '%s\n', num2str(ImgTS(i,:), '%g '));
    
    fprintf(fid3, '%s\n', num2str(ImgId(i,:), '%g '));
  end
  fclose(fid1);
  fclose(fid2);
  fclose(fid3);
  
  %% Write the imu data files
  fid4 = fopen('data/imu_data_true.dat','w+');
  fid5 = fopen('data/imu_data_meas.dat','w+');
  fid6 = fopen('data/pose_true.dat','w+');
  fid7 = fopen('data/attitude_meas.dat','w+');
  for i=1:N+1
    fprintf(fid4, '%s\n', num2str([Or(:,i)' Ar(:,i)' mod(i,100) 10*i], '%.14f '));
    fprintf(fid5, '%s\n', num2str([Om(:,i)' Am(:,i)' mod(i,100) 10*i], '%.14f '));
    fprintf(fid6, '%s\n', num2str([Qr(:,i)' Pr(:,i)' mod(i,100) 10*i], '%.14f '));
    fprintf(fid7, '%s\n', num2str([Qm(:,i)' 10*i], '%.14f '));
  end
  fclose(fid4);
  fclose(fid5);
  fclose(fid6);
  fclose(fid7);
  
  
  %% Write the state log file
  fid6 = fopen('data/state_true.dat','w+');
  for i=1:N %%changed N+1                                                                             !!!
    fprintf(fid6, '%s\n', num2str(xtk_log(:,i)', '%.14f '));
  end
  fclose(fid6);
  
  
  fid7 = fopen('data/map.dat', 'w+');
  fprintf(fid7, '%s\n', num2str(map.feats(1,1:map.nfeats_valid), '%.14f '));
  fprintf(fid7, '%s\n', num2str(map.feats(2,1:map.nfeats_valid), '%.14f '));
  fprintf(fid7, '%s\n', num2str(map.feats(3,1:map.nfeats_valid), '%.14f '));
  fclose(fid7);
  
  fid8 = fopen('data/IMU2Cam.dat', 'w+');
  fprintf(fid8, '%s ', num2str(IMU2Cam.q', '%.14f '));
  fprintf(fid8, '%s ', num2str(IMU2Cam.p', '%.14f '));
  fclose(fid8);
  
  fid9 = fopen('data/Earth.dat', 'w+');
  fprintf(fid9, '%s ', num2str(earth.g', '%.14f '));
  fprintf(fid9, '%s ', num2str(earth.omega', '%.14f '));
  fprintf(fid9, '%s ', num2str(earth.p', '%.14f '));
  fclose(fid9);
  
  fid10 = fopen('data/Sigma.dat', 'w+');
  fprintf(fid10, '%s ', num2str(sigma.r, '%.14f '));
  fprintf(fid10, '%s ', num2str(sigma.w, '%.14f '));
  fprintf(fid10, '%s ', num2str(sigma.g, '%.14f '));
  fprintf(fid10, '%s ', num2str(sigma.v, '%.14f '));
  fprintf(fid10, '%s ', num2str(sigma.px, '%.14f '));
  fprintf(fid10, '%s ', num2str(sigma.att, '%.14f '));
  fprintf(fid10, '%s ', num2str(sigma.gps, '%.14f '));
  fclose(fid10);
  
  
  fid11 = fopen('data/gps_data_meas.dat','w+');
  fid12 = fopen('data/gps_data_true.dat','w+');
  for i=1:size(GPS,2)
    fprintf(fid11, '%s ', num2str([GPS(:,i)' GPSTS(i)], '%.14f '));
    fprintf(fid12, '%s ', num2str([GPSm(:,i)' GPSTS(i)], '%.14f '));
  end
  fclose(fid11);
  fclose(fid12);
  
  
end
fprintf('Plotting figures ...\n');

PLOT_FIGURES

% ros_data_dir = '/home/sean/indigo_semslam_ws/code/cpp/src/simulator/data';
ros_data_dir = '/home/sean/code/object_pose_detection/src/semantic_slam/test/data/';

% write IMU

dlmwrite([ros_data_dir, '/times.dat'], imu_times, 'delimiter', ' ', 'precision', '%.8f');

dlmwrite([ros_data_dir, '/accel_real.dat'], Ar', 'delimiter', ' ', 'precision', '%.8f');
dlmwrite([ros_data_dir, '/accel_meas.dat'], Am', 'delimiter', ' ', 'precision', '%.8f');
dlmwrite([ros_data_dir, '/gyro_real.dat'], Or', 'delimiter', ' ', 'precision', '%.8f');
dlmwrite([ros_data_dir, '/gyro_meas.dat'], Om', 'delimiter', ' ', 'precision', '%.8f');

% write camera
WRITE_CAMERA = false;

if WRITE_CAMERA
    imgr_file = fopen([ros_data_dir, '/cam_real.dat'], 'w');
    imgm_file = fopen([ros_data_dir, '/cam_meas.dat'], 'w');

    % format: 
    % TIME
    % N_MSMTS
    % FEAT_ID MSMT FEAT_ID MSMT ... FEAT_ID MSMT
    %
    % example image at time 30 that observes fts 31 and 53:
    % 30
    % 2
    % 31 0.3190 0.0135 53 -0.14 0.68

    for i = 1:size(ImgPtsU, 1)
        fprintf(imgr_file, '%d\n', ImgTS(i));
        fprintf(imgr_file, '%d\n', map.nfeats_per_im);
        fprintf(imgm_file, '%d\n', ImgTS(i));
        fprintf(imgm_file, '%d\n', map.nfeats_per_im);
        for j = 1:map.nfeats_per_im
            % "real" (noiseless)
            ur = cam.f * ImgPtsU(i,j) + cam.c(1);
            vr = cam.f * ImgPtsV(i,j) + cam.c(2);
            fprintf(imgr_file, '%d %.8f %.8f ', ImgId(i,j), ur, vr);

            % measured (w/ noise)
            um = cam.f * ImgPtsUm(i,j) + cam.c(1);
            vm = cam.f * ImgPtsVm(i,j) + cam.c(2);
            fprintf(imgm_file, '%d %.8f %.8f ', ImgId(i,j), um, vm);
        end
        fprintf(imgr_file, '\n');
        fprintf(imgm_file, '\n');
    end
end

return
