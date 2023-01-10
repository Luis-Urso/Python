import math

def calculate_circle_area(radius):
    area = math.pi * radius * radius
    return area

circle_radius = 5
area = calculate_circle_area(circle_radius)
print("The area of the circle with radius", circle_radius, "is", area)