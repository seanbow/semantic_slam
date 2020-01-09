% SCRIPT: plot the simulation figures
%
% $Id: PLOT_FIGURES.m 1044 2011-03-17 13:16:23Z joel $
% $LastChangedDate: 2011-03-17 08:16:23 -0500 (Thu, 17 Mar 2011) $
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


figure(); subplot(4,1,1); plot(Qr(1,:)); subplot(4,1,2); plot(Qr(2,:)); 
subplot(4,1,3); plot(Qr(3,:)); subplot(4,1,4); plot(Qr(4,:));
title('Real Quaternion');

figure(); subplot(3,1,1); plot(Pr(1,:)); subplot(3,1,2); plot(Pr(2,:)); 
subplot(3,1,3); plot(Pr(3,:));
title('Real Quaternion');

figure(); plot3(Pr(1,:),Pr(2,:),Pr(3,:)); hold on; 
plot3(pxatt,pyatt,pzatt,'.'); grid on;
title('Real Trajectory');
%% Add the map
% for i=1:9 %length(map)
%     if map(i).v'*e1 == 0
%         plot3(0:50:50, [map(i).v(2) map(i).v(2)], [0 0], 'r--');
%     elseif map(i).v'*e2 == 0
%         plot3([map(i).v(1) map(i).v(1)], 0:30:30, [0 0], 'r--');
%     end
% end

%% Add the map

for i = 1:map.nfeats_valid
  plot3(map.feats(1,i), map.feats(2,i), map.feats(3,i), 'rx')
end