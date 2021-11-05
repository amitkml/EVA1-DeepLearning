Calculations:
------------------
In accordance with the block diagram i calculated 
for the first network with 2 FC and softmax 
that is in page no 7


Layer  convolution 1 :
---------

n_out = (224 + (2*1)-7) /2 +1 
      = 222/2 + 1 =112

j_in = 1*2=2


r_out = 1 +(7-1)(1*1)
      = 7

 Max pool :
----------
n_out = (112 + (2*0) -3)/2 +1
      = 109 /2 + 1 = 56

j_in = 2*2 = 4

r_out = 7+(3-1)*(1*2)
      = 11 

layer  convolution 2:
-----------------

n_out = 56 + 2(0) -3 /1 + 1 
      = 56

j_in = 4*1 = 4

r_out = 11 + (3-1)(1*4)
      = 19

 Max pool :
-----------
n_out = 56 + 2(0) -3 /2 + 1 =28

j_in = 4 * 2 = 8

r_out = 19 + (3-1)(1*4)
      = 27

layer - inception 3a
--------- 
n_out = 28 + 2(2) -5 /1 + 1 = 28

j_in = 1* 8 = 8

r_out = 27 + (5-1)*(1*8)
      = 59

layer  - inception 3b
------------
n_out = 28 + 2(2) -5 /1 + 1 = 28

j_in = 1* 8 = 8

r_out = 59 + (5-1)*(1*8)
      = 91

Maxpool :
----------

n_out = 28 + 2(2) -5 /2 + 1 = 14

j_in = 8* 2 = 16

r_out = 91 + (3-1)(1*8)
      = 97

layer 4a:
------------

n_out = 24 + 2(2) -5 /1 + 1 = 14
j_in = 16*1 = 16
r_out = 97 + (5-1)(1*16)
      = 161

Average pool 
-------------
n_out = 14 + 2(3) - 5 /2 = 7
j_in = 16*3 = 48
r_out = 161 + (5-1)(1*16)
      = 225

#--------------------------------------






















































