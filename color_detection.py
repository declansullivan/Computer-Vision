"""
OPENCV Color Detection for Triton Robosub
Authored by Declan Sullivan
"""

import cv2
import numpy as np

# COLOR FILTERING

# Webcam / Constants
capture = cv2.VideoCapture(0)
switch = '1: bars'
_, frame = capture.read()

# frame = cv2.imread('testImages/picture4.png')
# frame = cv2.resize(frame, (1000, 562))

height, width, channels = frame.shape
midframe = (width // 2, height // 2)

# RED #
# lower = np.array([10, 10, 100])
# upper = np.array([100, 130, 255])

# SLIDER BASED RED VALUES
# lower = np.array([0, 0, 25])
# upper = np.array([50, 40, 255])

# BLUE #
# lower = np.array([100, 10, 10])
# upper = np.array([220, 50, 50])

# ORANGE #
# HSV #
lower = np.array([0, 115, 147])
upper = np.array([17, 231, 255])

# BU - 17
# GU - 231
# RU - 255

# BL - 0
# GL - 115
# RL - 147

# BGR #
# lower = np.array([0, 20, 75])
# upper = np.array([40, 160, 255])

# TEMP PICTURE4 BGR #
# lower = np.array([73, 37, 0])
# upper = np.array([82, 92, 232])


# This area can be used to filter out the smaller portions of orange that
# might throw off the algorithm.
min_orange_area = 100

# This threshold can be used to determine how accurately you want to robot
# to be centered with the middle of the gate.
min_gate_threshold_ratio = 25


def distance(p1, p2):
    """Return distance between two points.
    @input p1: First point (x, y)
    @input p2: Second point (x, y)
    @return Returns the distance between two points using the 
            distance formula.
    """
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def midpoint_form(p1, p2):
    """Return midpoint between two points.
    @input p1: First point (x, y)
    @input p2: Second point (x, y)
    @return Returns the midpoint of these two points, using integer 
            division to avoid bugs with opencv.
    """
    return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)

def nothing(x):
    """Default method required to make sliders function properly
    with opencv.
    """
    pass


def create_bars():
    """Generates a window with sliders. The sliders are used to
    manually enter in color values so that it is easier to find
    color thresholds for different situations.
    """

    # Window to contain bars.
    cv2.namedWindow('bars')

    # Upper bounds of color bars.
    cv2.createTrackbar('BU','bars',0,255,nothing)
    cv2.createTrackbar('GU','bars',0,255,nothing)
    cv2.createTrackbar('RU','bars',0,255,nothing)

    # Lower bounds of color bars.
    cv2.createTrackbar('BL','bars',0,255,nothing)
    cv2.createTrackbar('GL','bars',0,255,nothing)
    cv2.createTrackbar('RL','bars',0,255,nothing)

    # Creates a trackbar that makes the above ranges toggleable.
    cv2.createTrackbar(switch,'bars',0,1,nothing)

def build_triangle(midpoint, image):
    """Build triangle to align robot with center of gate.
    @input midpoint: The midpoint of the two largest circles that 
                     are closest together, which is the center of
                     the smaller gate.
    @input image:   The current frame being analyzed so that the 
                    triangle and other relevant information can be 
                  displayed on it.
    @return Returns vert_leg_point, which is the point on the 
            triangle that is connected to the midpoint of the 
            gate and the center of the frame.
    """
    triange_hypotenuse = distance(midframe, midpoint)           # Distance between midpoint of circles and center of frame.
    vert_leg_point = (midframe[0], midpoint[1])                 # Point above/below center of frame with same y-coordinate of midpoint of circles.
    triangle_vert_leg = distance(midframe, vert_leg_point)      # Length of vertical leg of triangle.
    triangle_horiz_leg = distance(midpoint, vert_leg_point)     # Length of horizontal leg of triangle.
    cv2.line(image, midframe, vert_leg_point, (0,255,0),2)      # Draw vertical leg.
    cv2.line(image, midpoint, vert_leg_point, (0,255,0),2)      # Draw horizontal leg.
    return vert_leg_point


def direction_recommender(midpoint):
    """Based on point between two largest, closest circle, determine direction
    to move the robot to bring that point to center of the screen.
    @input midpoint: The midpoint of the two largest circles that
                     are closest together, which is the center of
                     the smaller gate.
    @return Returns nothing but prints out the possible instructions
            that could be sent to state machine. This can be altered
            to have actual return values or send the output to a file,
            however is best for state machine.
    """
    up_down = 0
    left_right = 0

    # Stay still if your ratio of gate to center of frame is less than defined.
    if abs(midpoint[0] - midframe[0]) <= min_gate_threshold_ratio: pass
    else:
        # If you are not accurately aligned, move left or right until you are.
        if midpoint[0] > midframe[0]: up_down = 1
        else: up_down = -1

    # Stay still if your ratio of gate to center of frame is less than defined.
    if abs(midpoint[1] - midframe[1]) <= min_gate_threshold_ratio: pass
    else:
        # If you are not accurately aligned, move left or right until you are.
        if midpoint[1] < midframe[1]: left_right = 1
        else: left_right = -1
            
    print((left_right, up_down))


def generate_mask(hsv):
    """Generate a mask based on threshold values.
    @input hsv: The current image being analyzed, in hsv color values.
    @return Returns a boolean image, a mask, showing areas of the image
            that match the given threshold. This is what is used to filter
            out colors that are not orange.
    """
    # Get value of the switch to determine whether or not to use the bar window.
    s = cv2.getTrackbarPos(switch, 'bars')

    if s:
        # Get upper bounded values from trackbar window.
        bu = cv2.getTrackbarPos('BU', 'bars')
        gu = cv2.getTrackbarPos('GU', 'bars')
        ru = cv2.getTrackbarPos('RU', 'bars')

        # Get lower bounded values from trackbar window.
        bl = cv2.getTrackbarPos('BL', 'bars')
        gl = cv2.getTrackbarPos('GL', 'bars')
        rl = cv2.getTrackbarPos('RL', 'bars')

        # Construct usable arrays using upper and lower bounds.
        user_lower = np.array([bl, gl, rl])
        user_upper = np.array([bu, gu, ru])

        # Filter out colors.
        return cv2.inRange(hsv, user_lower, user_upper)
    else:
        # Filter out colors.
        return cv2.inRange(hsv, lower, upper)


def find_gates(frame, circles):
    """Takes in circles, which are all of the possible areas in the frame
    that a gate post was found.
    @input frame: Current image to draw to and analyze.
    @input circles: List of areas that could be a gate post.
    """
    # If you have found circles, then there are possible gate posts to inspect.
    if len(circles) > 0:
        circles.sort(key=lambda x: x[1], reverse=True)                      # Pick circles with the three largest radii.
        circles = circles[:3]                                               # Delete extra circles
        
        for i in circles:
            cv2.circle(frame, i[0], i[1], (0,255,0),2)                      # Draw each circle to screen.

        if len(circles) == 2:
            midpoint = midpoint_form(circles[0][0], circles[1][0])          # Find midpoint between two circles found.
            cv2.line(frame, circles[0][0], circles[1][0], (0,255,0), 2)     # Draw line connecting radii.
            cv2.line(frame, midframe, midpoint, (0,255,0),2)                # Draw line connecting midpoint and frame.
            cv2.circle(frame, midpoint, 10, (0,0,255),-1,2)                 # Draw red dot at midpoint.

            # Trig for determining angle above/below
            vert_leg_point = build_triangle(midpoint, frame)                # Create triangle for visual reference.
            direction_recommender(midpoint)                                 # Choose directions to move based on midpoint of circles.

        elif len(circles) == 3:
            # Need to compare the three largest circles' distances to find the ones closest together.
            circle_comparisons = [(0, 1), (1, 2), (0, 2)]

            # circle_info contains the distance between two circles, and the circles themselves.
            circle_info = []

            for pair in circle_comparisons:
                first_circle = circles[pair[0]]
                second_circle = circles[pair[1]]

                # Append the distance between the circles, and the circles' center points.
                circle_info.append((distance(first_circle[0], second_circle[0]), first_circle[0], second_circle[0]))

            smallest_distance = min(circle_info, key=lambda x: x[0])            # Find index of circles that are closest together.
            closest_posts = circle_info[circle_info.index(smallest_distance)]   # Find full info on closest circles.
            midpoint = midpoint_form(closest_posts[1], closest_posts[2])        # Find midpoint between closes circles.

            cv2.line(frame, closest_posts[1], closest_posts[2], (0,255,0),2)    # Draw line between circle centers.
            cv2.line(frame, midframe, midpoint, (0,255,0),2)                    # Draw line between midpoint and center of frame.
            cv2.circle(frame, midpoint, 10, (0,0,255),-1,2)                     # Draw red circle at midpoint.

            # Trig for determining angle above/below
            vert_leg_point = build_triangle(midpoint, frame)                    # Draw the triangle to find angles/distances.
            direction_recommender(midpoint)                                     # Recommend movement for robot.

        cv2.circle(frame, midframe, 10, (0,0,255),-1,2)                         # Draw red circle in middle of frame.



def find_contours(frame, contours):
    """Takes in contours, areas of the image with similar color/intensity, and 
    looks for areas in the image that could be parts of the gate.
    @input frame: Current image to draw to and analyze.
    @input contours: List of areas that have similar color/intensity.
    """
    # If there are contours, work with them, otherwise no gate was found.
    # This can be expanded on to inform the robot to continue looking for gates.
    if len(contours) > 0:
        circles = []
        
        # Iterate through each contour to determine whether or not it is valuable.
        for c in contours:

            # Get the area of the current contour.
            area = cv2.contourArea(c) 

            # Find contour spaces that have a large enough area to be considered significant.
            if area > min_orange_area:
                # Define values for circle that encloses the found gate post.
                (x, y), radius = cv2.minEnclosingCircle(c)
                center = (int(x), int(y))
                radius = int(radius)

                # Add circle to list, this is a possible gate post.
                # Each circle is a tuple containing its center and radius.
                circles.append((center, radius))

        # Find out which circles could be gate posts.
        find_gates(frame, circles)
        


def main():
    """Main method. Runs the while loop that continuously performs image
    analysis until the user commands it to stop. Each frame is converted
    to hsv color, then sent off to the find_contours method, which performs
    the rest of the analysis.
    
    The workflow of each frame is as follows:
    1. Read in a new frame (image) from the webcam.
    2. Blur the image to make colors easier to identify.
    3. Convert the image to hsv values, as opposed to bgr.
    4. Create a mask, which filters out colors other than the ones you want.
    5. Remove areas containing illegal color from original image.
    6. Remove areas in the mask that may be a shade of gray.
    7. Find the contours, similar areas, of the mask.
    8. Use these contours to find gate posts.
    """
    while True:
        # Get next frame to analyze.
        _, frame = capture.read()

        # Use test image instead.
        # frame = cv2.imread('testImages/picture4.png')
        # frame = cv2.resize(frame, (1000, 562))
        # height, width, channels = frame.shape
        # midframe = (width // 2, height // 2)

        # Blur the image to make color detection simpler.
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)

        # Convert frame to hsv values
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        # Generate mask based on lower and upper BGR bounds
        mask = generate_mask(hsv)
        
        # Generate result based on mask.
        res = cv2.bitwise_and(frame, frame, mask=mask)

        # Threshold for white based on mask.
        retval, threshold = cv2.threshold(mask, 20, 255, cv2.THRESH_BINARY)

        # Finds local areas of the image that have the same color/intensity.
        contours, hierarchy = cv2.findContours(threshold, 1, 2)
        find_contours(frame, contours)

        cv2.imshow('live', frame)               # Show current frame
        cv2.imshow('mask', mask)              # Show boolean mask
        # cv2.imshow('res', res)                # Show filtered color
        # cv2.imshow('threshold', threshold)    # Show thresholded values

        # Press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    create_bars()
    main()

    cv2.destroyAllWindows()
    capture.release()
