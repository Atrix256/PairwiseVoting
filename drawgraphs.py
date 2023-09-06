import math
from PIL import Image, ImageDraw, ImageFont

w, h = 2048, 2048
outputw, outputh = 512, 512
dotRadius = w * 0.01
circleRadius = w * 0.38
circleTextRadius = w * 0.45
lineWidth = w * 0.01

diagramSettingsList =[
    {
        "fileName": "5.png",
        "numPoints" : 5,
        "links" : [
            [0, 1, "red"],
            [1, 2, "red"],
            [2, 3, "red"],
            [3, 4, "red"],
            [4, 0, "red"],

            [1, 3, "green"],
            [2, 4, "green"],
            [3, 0, "green"],
            [4, 1, "green"],
            [0, 2, "green"],
        ]
    },
    {
        "fileName": "6.png",
        "numPoints" : 6,
        "links" : [
            [0, 1, "red"],
            [1, 2, "red"],
            [2, 3, "red"],
            [3, 4, "red"],
            [4, 5, "red"],
            [5, 0, "red"],

            [1, 3, "green"],
            [2, 4, "green"],
            [3, 5, "green"],
            [4, 0, "green"],
            [5, 1, "green"],
            [0, 2, "green"],

            #[0, 3, "grey"],
            #[1, 4, "grey"],
            #[2, 5, "grey"],
        ]
    },
    {
        "fileName": "7.png",
        "numPoints" : 7,
        "links" : [
            [0, 1, "red"],
            [1, 2, "red"],
            [2, 3, "red"],
            [3, 4, "red"],
            [4, 5, "red"],
            [5, 6, "red"],
            [6, 0, "red"],

            [1, 3, "green"],
            [2, 4, "green"],
            [3, 5, "green"],
            [4, 6, "green"],
            [5, 0, "green"],
            [6, 1, "green"],
            [0, 2, "green"],

            [3, 6, "blue"],
            [4, 0, "blue"],
            [5, 1, "blue"],
            [6, 2, "blue"],
            [0, 3, "blue"],
            [1, 4, "blue"],
            [2, 5, "blue"],            
        ]
    },
]

font = ImageFont.truetype("arial.ttf", 150)

def TextPointLocation(index, count):
    angle = index / count * math.pi * 2.0
    angle = angle - math.pi / 2    
    return [math.cos(angle) * circleTextRadius + w / 2.0, math.sin(angle) * circleTextRadius + h / 2.0]

def PointLocation(index, count):
    angle = index / count * math.pi * 2.0
    angle = angle - math.pi / 2    
    return [math.cos(angle) * circleRadius + w / 2.0, math.sin(angle) * circleRadius + h / 2.0]

for settings in diagramSettingsList:
    img = Image.new("RGB", (w, h))
    img1 = ImageDraw.Draw(img)  

    img1.rectangle([0, 0, w, h], fill="white")

    for pointIndex in range(settings["numPoints"]):
        location = PointLocation(pointIndex, settings["numPoints"])
        dotBox = [(location[0] - dotRadius, location[1] - dotRadius), (location[0] + dotRadius, location[1] + dotRadius)]
        img1.ellipse(dotBox, fill="black")

        location = TextPointLocation(pointIndex, settings["numPoints"])
        textbb = img1.textbbox([0,0], str(pointIndex), font=font)
        half = [(textbb[2] - textbb[0])/2, (textbb[3] - textbb[1])/2]
        img1.text([location[0] - half[0], location[1] - half[1] - textbb[1]], str(pointIndex), font = font, align ="center", fill="black") 

    for link in settings["links"]:
        location1 = PointLocation(link[0], settings["numPoints"])
        location2 = PointLocation(link[1], settings["numPoints"])
        img1.line([location1[0], location1[1], location2[0], location2[1]], fill=link[2], width=int(lineWidth))

    img.resize([outputw, outputh]).save(settings["fileName"])
