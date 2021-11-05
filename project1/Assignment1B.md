1.What are Channels and Kernels ?

    Channels - Channels are number of representationalformat/colors/features in a image eg: RGB represents level of red,green and blue colour in a pixel/image.
    take a  image of 28x28x3 which has 3 channels and apply 58 different kernels you can 
    have 58 different features/channels.


    Kernels - Kernels is a matrix or a set of numbers that is used 
    to find or apply  a specific pattern in a image .
    eg: edge detection - performed by  identifying 
    the change in brightness of a image

2.Why should we only use 3x3 kernels ?

**Usage of odd kernels** - 
*symentric kernels -*

    Convolution is performed on a pixel (source) and its surroding pixel .So if we keep the source
    as anchor point then we need to have equal no of pixels on all 4 sides of anchor pixel.

    formula - 2*n+1 
    Hence , if one pixel(n=1) on all sides then --> 3x3
                1   1  1
                1   sp 1
                1   1  1  

    or  if two pixel(n=2) on all side  then --> 5x5 

            1   1   1   1   1
            1   1   1   1   1
            1   1   sp  1   1
            1   1   1   1   1
            1   1   1   1   1


*Asymentric Kernels*

    Similarly even if you use asymentric kernels (3x1 or 1x3 or 7x1 or anyother )you would observe that only odd kernel could be applied for
    convolution process properly to modify each pixel

**Usage of 3x3 kernels -**

    With 3x3 we can make any type of convolutional result  that is performed by higher level of kernels . eg: if you use two 3x3 over a image the result will be same as the output of one 5x5 kernel . i.e two 3x3 gives a receptive field of one 5x5 kernel.


3.How many times do we need to perform 3x3 convolution 
  operation to reach 1x1 from 199x199 ?'

  We need to perform 3x3 conv 99 times to reach 1x1 from 199x199.

    199x199 -> 197x197 --> 195x195 --> 193x193 --> 191x191 --> 189x189 -->
    187x187 -> 185x185 --> 183x183 --> 181x181 --> 179x179 --> 177x177 -->
    175x175 -> 173x173 --> 171x171 --> 169x169 --> 167x167 --> 165x165 -->
    163x163 -> 161x161 --> 159x159 --> 157x157 --> 155x155 --> 153x153 -->
    151x151 -> 149x149 --> 147x147 --> 145x145 --> 143x143 --> 141x141 -->
    139x139 -> 137x137 --> 135x135 --> 133x133 --> 131x131 --> 129x129 -->
    127x127 -> 125x125 --> 123x123 --> 121x121 --> 119x119 --> 117x177 -->
    115x115 -> 113x113 --> 111x111 --> 109x109 --> 107x107 --> 105x105 -->
    103x103 -> 101x101 --> 99x99   --> 97x97   --> 95x95   --> 93x93   -->
    91x91   -> 89x89   --> 87x87   --> 85x85   --> 83x83   --> 81x81   -->
    79x79   -> 77x77   --> 75x75   --> 73x73   --> 71x71   --> 69x69   -->
    67x67   -> 65x65   --> 63x63   --> 61x61   --> 59x59   --> 57x57   -->
    55x55   -> 53x53   --> 51x51   --> 49x49   --> 47x47   --> 45x45   -->
    43x43   -> 41x41   --> 39x39   --> 37x37   --> 35x35   --> 33x33   -->
    31x31   -> 29x29   --> 27x27   --> 25x25   --> 23x23   --> 21x21   -->
    19x19   -> 17x17   --> 15x15   --> 13x13   --> 11x11   --> 9x9     -->
    7x7     -> 5x5     --> 3x3     --> 1x1














