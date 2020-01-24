function [ppout,static_intervals]= PoseGen1D(t,px)
% 1 Dimensional Pose Generator Using Spline
%
% $Id: PoseGen1D.m 1044 2011-03-17 13:16:23Z joel $
% $LastChangedDate: 2011-03-17 08:16:23 -0500 (Thu, 17 Mar 2011) $
% Faraz Mirzaei
% faraz@cs.umn.edu

% clc
% clear
% close all
% 
% 
% 
% px = [1 1 1 2 6 7 7 8];
% t = 1:length(px);
% tfine = 1:0.01:length(px);

if length(t) ~= length(px)
    error('t and px must have the same dimension!')
end

dpx = px - circshift(px,[0 -1]);
dpx(end) = inf;
const_periods = find(~logical(dpx));

if isempty(const_periods)
    intervals = [1 length(px) 3];    
elseif const_periods(1) ~= 1
    intervals(1,:) = [1 const_periods(1) 3];
else
    intervals = [];
end

for k = 1:length(const_periods)
    intervals = [intervals ; const_periods(k) const_periods(k)+1 1];
    if (length(px) > const_periods(k)+1) 
        if (k < length(const_periods))
            intervals = [intervals ; const_periods(k)+1 const_periods(k+1) 3];
        else
            intervals = [intervals ; const_periods(k)+1 length(px) 3];
        end
    end
end

ind_rem = [];
for k = 1:size(intervals,1)
    if intervals(k,1) == intervals(k,2)
        ind_rem = [ind_rem k];
    end
end
intervals(ind_rem,:) = [];

% intervals
pp = [];
static_intervals = [];
for k = 1:size(intervals,1)
    ptr1 = intervals(k,1); ptr2 = intervals(k,2);
    if intervals(k,3) == 1
        pp_tmp = interp1([t(ptr1) t(ptr2)],[px(ptr1) px(ptr2)],'linear','pp');
        static_intervals = [static_intervals [t(ptr1) ; t(ptr2)]];
    elseif intervals(k,3) == 3
        ch_tmp = spapi(4,[t(ptr1:ptr2) t(ptr1) t(ptr2) t(ptr1) t(ptr2)], ...
            [px(ptr1:ptr2) 0 0 0 0]);
        pp_tmp = fn2fm(ch_tmp,'pp');
    else
        error('Undefined Spline order!');
    end

    pp = ppcat(pp,pp_tmp);
    
end
    
ppout = pp;

% figure();
% plot(tfine,fnval(pp,tfine)); grid
% 
% figure();
% plot(tfine,fnval(fnder(pp),tfine)); grid
    
    