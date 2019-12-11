s/List//g
s/p([0-9])/p(\1)/g
s/q([ab]*)0/q\1.x()/g
s/q([ab]*)1/q\1.y()/g
s/q([ab]*)2/q\1.z()/g
s/q([ab]*)3/q\1.w()/g
s/,/,\n/g