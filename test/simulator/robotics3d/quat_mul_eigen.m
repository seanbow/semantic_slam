function q = quat_mul_eigen(a, b)

q = zeros(4,1);

q(1) = a(4) * b(1) + a(1) * b(4) + a(2) * b(3) - a(3) * b(2);
q(2) = a(4) * b(2) + a(2) * b(4) + a(3) * b(1) - a(1) * b(3);
q(3) = a(4) * b(3) + a(3) * b(4) + a(1) * b(2) - a(2) * b(1);
q(4) = a(4) * b(4) - a(1) * b(1) - a(2) * b(2) - a(3) * b(3);

q = q/norm(q);