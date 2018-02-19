# meters
Meter reading via web cam.

Inspired by https://www.mkompf.com/cplus/emeocv.html

## The hot water meter
The hot water measures the flow of hot water into the house.
It has a mechanical counter for cubic meters, plus dials
for the less significant digits.

My goal was to be able to read the value of the meter from
a single picture. For the purpose of monitoring the water
usage a simpler approach of just counting rotations of the
fastest moving dial would work, but this project is just as
much about learning something about image processing with
OpenCV as it is about getting raw data from the meter.

1. Locate digits
2. Extract digit values
3. Locate dials
4. Extract dial values

### Locating digits
Threshold the image, to get a sharp black and white image, then
find contours. There will be lots of contours found, the trick
is to find the ones that represent the digits. I know the
approximate size so I first discard ones that are too small or
too large. I know that the digits are all lined up so I look
for a chain of contours that have the same approximate vertical
location.

### Extracting digit values
Once I've found the digits on the meter, I extract their image
as a 16x16 grayscale image and use the k-Nearest Neighbor approach
to identify the digit. For this to work, I first need some training
data so I simply let the program run for a while, saving out the
digit images. I then manually looked at those images, sorted them
into folders for each digit and used that as the training data.
Once the kNN algorithm has seen several samples of each digit it
is quite reliable in recognizing them.

### Locating dials
To locate the dials I make use of the fact that I know their
relative position to the digits, so once the digits have been
located, the search for the dials is restricted to a fixed area
below them. Then I use a similar approach of looking for contours
in the thresholded image, filtering out contours that are too
big or too small. In addition to the dials there is a rotating
wheel, I guess to show that water is flowing even if it is very slow.
That wheel needed some extra filtering as it is of a similar size
as the dials, but I can use the fact that it is symmetrical whereas
the dials are always longer in one dimension.

### Extracting dial values
The contours for the dials represent their shape quite nicely.
To get their value, I fit a convex hull around the contour, then
fit a line through those points. The angle of the line then gives
me the angle of the dial, making it straight forward to derive
the digit value. The only caveat is that the line fitting only
seems to give values for half the circle - it doesn't pay attention
to which direction the needle is pointing so it may give values
that are off by 180 degrees.

To work around that, I determine which direction the needle is pointing
by extracting the pixels for the dial as a rotated image, using the
angle determined from the line. If the resulting image is wider than it
is high, I split into left/right halves and determine which half is
darker. The darker half is where the needle is, the lighter half has
the center of the dial. If the image is taller than it is wide, I split
into top/bottom halves.