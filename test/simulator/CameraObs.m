function [img, map] = CameraObs(xtk, sigma, map, IMU2Cam)



img.feats = zeros(2,map.nfeats_per_im);
img.feats_meas = zeros(2,map.nfeats_per_im);
img.feat_id = zeros(1,map.nfeats_per_im);

% global to imu
g_P_i = xtk(14:16,1);
g_C_i = quat2rot_eigen(xtk(1:4,1))';

% i_P_g = - g_C_i' * g_P_i;

% imu to laser
i_P_c = IMU2Cam.p;
i_C_c = quat2rot_eigen(IMU2Cam.q);

p_in_Cam = i_C_c' * g_C_i' * ( map.feats(:,1:map.nfeats_valid) - repmat(g_P_i, 1, map.nfeats_valid) - g_C_i * repmat(i_P_c, 1, map.nfeats_valid));

jpt = 1;
for i = 1:map.nfeats_valid
  theta = acos(p_in_Cam(3,i)/norm(p_in_Cam(:,i))) * 180/pi;
  
  
  if (abs(theta) < 22.5) && (p_in_Cam(3,i) > 1) % 45 deg fov && at least 1 meter in front of the camera
    img.feats(:,jpt) = [p_in_Cam(1,i)/p_in_Cam(3,i) ; p_in_Cam(2,i)/p_in_Cam(3,i)];
    img.feats_meas(:,jpt) = img.feats(:,jpt) + sigma.px * randn(2,1);
    img.feat_id(jpt) = i;
    jpt = jpt + 1;
  end
  
  if jpt > map.nfeats_per_im
    break;
  end
  
end

if jpt <= map.nfeats_per_im
  
  for k = jpt:map.nfeats_per_im
    % randomly generate a new point
    
    theta = (45 * rand - 22.5) * pi/180;
    phi = (45 * rand - 22.5) * pi/180;
    
    punit = roty(theta) * rotx(phi) * [0;0;1];
    
    p_in_Cam = (20 * rand + 5) * punit;
    
    map.nfeats_valid = map.nfeats_valid + 1;
    map.feats(:,map.nfeats_valid) = g_P_i + g_C_i * i_P_c + g_C_i * i_C_c * p_in_Cam;
    
    img.feats(:,k) = [p_in_Cam(1)/p_in_Cam(3) ; p_in_Cam(2)/p_in_Cam(3)];
    img.feats_meas(:,k) = img.feats(:,k) + sigma.px * randn(2,1);
        
    img.feat_id(k) = map.nfeats_valid;
    
  end
  
end