function ppout = ppcat(pp1,pp2)
% Polynomial Concatanation Function (in pp form)
%
% $Id: ppcat.m 1044 2011-03-17 13:16:23Z joel $
% $LastChangedDate: 2011-03-17 08:16:23 -0500 (Thu, 17 Mar 2011) $
% Faraz Mirzaei
% faraz@cs.umn.edu

if isempty(pp1)
    ppout = pp2;
    return
elseif isempty(pp2)
    ppout = pp1;
    return
end


if ~(strcmp(pp1.form,'pp') && strcmp(pp2.form,'pp'))
    error('Only pp form is supported!')
end

if pp1.breaks(end) ~= pp2.breaks(1)
    error('pp1 and pp2 must share the same boundary!');
end

if pp1.dim ~= pp2.dim
    error('pp1 and pp2 must have the same dimension!');
end

ppout.form = 'pp';
ppout.breaks = [pp1.breaks(1:end) pp2.breaks(2:end)];
ppout.order = max(pp1.order,pp2.order);
ppout.dim = pp1.dim;
ppout.pieces = pp1.pieces + pp2.pieces;

if pp1.order < pp2.order
   tmpcoefs = [zeros(pp1.pieces,pp2.order-pp1.order) pp1.coefs];
   ppout.coefs = [tmpcoefs ; pp2.coefs];
else
   tmpcoefs = [zeros(pp2.pieces,pp1.order-pp2.order) pp2.coefs];
   ppout.coefs = [pp1.coefs ; tmpcoefs];
end



