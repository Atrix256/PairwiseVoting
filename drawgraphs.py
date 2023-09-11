import math
from PIL import Image, ImageDraw, ImageFont

w, h = 2048, 2048
outputw, outputh = 256, 256
dotRadius = w * 0.01
circleRadius = w * 0.35
circleTextRadius = w * 0.42
circleVerticalOffset = w * 0.05
lineWidth = w * 0.01

diagramSettingsList =[
    {
        "fileName": "5Add1.png",
        "title" : "5 Nodes, Cycle 1",
        "numPoints" : 5,
        "links" : [
            [0, 1, "red"],
            [1, 2, "red"],
            [2, 3, "red"],
            [3, 4, "red"],
            [4, 0, "red"]
        ]
    },
    {
        "fileName": "5Add2.png",
        "title" : "5 Nodes, Cycle 1 and 2",
        "numPoints" : 5,
        "links" : [
            [0, 1, "red"],
            [1, 2, "red"],
            [2, 3, "red"],
            [3, 4, "red"],
            [4, 0, "red"],

            [0, 2, "green"],
            [1, 3, "green"],
            [2, 4, "green"],
            [3, 0, "green"],
            [4, 1, "green"],
        ]
    },
    {
        "fileName": "5Add3.png",
        "title" : "5 Nodes, Cycle 3",
        "numPoints" : 5,
        "links" : [
            [0, 3, "blue"],
            [1, 4, "blue"],
            [2, 0, "blue"],
            [3, 1, "blue"],
            [4, 2, "blue"],
        ]
    },
    {
        "fileName": "13Add1.png",
        "title" : "13 Nodes, Cycle 1",
        "numPoints" : 13,
        "links" : [
            [0, 1, "red"],
            [1, 2, "red"],
            [2, 3, "red"],
            [3, 4, "red"],
            [4, 5, "red"],
            [5, 6, "red"],
            [6, 7, "red"],
            [7, 8, "red"],
            [8, 9, "red"],
            [9, 10, "red"],
            [10, 11, "red"],
            [11, 12, "red"],
            [12, 0, "red"],
        ]
    },
    {
        "fileName": "13Add2.png",
        "title" : "13 Nodes, Cycle 2",
        "numPoints" : 13,
        "links" : [
            [0, 2, "green"],
            [1, 3, "green"],
            [2, 4, "green"],
            [3, 5, "green"],
            [4, 6, "green"],
            [5, 7, "green"],
            [6, 8, "green"],
            [7, 9, "green"],
            [8, 10, "green"],
            [9, 11, "green"],
            [10, 12, "green"],
            [11, 0, "green"],
            [12, 1, "green"]
        ]
    },
    {
        "fileName": "13Add3.png",
        "title" : "13 Nodes, Cycle 3",
        "numPoints" : 13,
        "links" : [
            [0, 3, "blue"],
            [1, 4, "blue"],
            [2, 5, "blue"],
            [3, 6, "blue"],
            [4, 7, "blue"],
            [5, 8, "blue"],
            [6, 9, "blue"],
            [7, 10, "blue"],
            [8, 11, "blue"],
            [9, 12, "blue"],
            [10, 0, "blue"],
            [11, 1, "blue"],
            [12, 2, "blue"]
        ]
    },
    {
        "fileName": "13Add4.png",
        "title" : "13 Nodes, Cycle 4",
        "numPoints" : 13,
        "links" : [
            [0, 4, "yellow"],
            [1, 5, "yellow"],
            [2, 6, "yellow"],
            [3, 7, "yellow"],
            [4, 8, "yellow"],
            [5, 9, "yellow"],
            [6, 10, "yellow"],
            [7, 11, "yellow"],
            [8, 12, "yellow"],
            [9, 0, "yellow"],
            [10, 1, "yellow"],
            [11, 2, "yellow"],
            [12, 3, "yellow"]
        ]
    },
    {
        "fileName": "13Add5.png",
        "title" : "13 Nodes, Cycle 5",
        "numPoints" : 13,
        "links" : [
            [0, 5, "purple"],
            [1, 6, "purple"],
            [2, 7, "purple"],
            [3, 8, "purple"],
            [4, 9, "purple"],
            [5, 10, "purple"],
            [6, 11, "purple"],
            [7, 12, "purple"],
            [8, 0, "purple"],
            [9, 1, "purple"],
            [10, 2, "purple"],
            [11, 3, "purple"],
            [12, 4, "purple"]
        ]
    },
    {
        "fileName": "13Add6.png",
        "title" : "13 Nodes, Cycle 6",
        "numPoints" : 13,
        "links" : [
            [0, 6, "orange"],
            [1, 7, "orange"],
            [2, 8, "orange"],
            [3, 9, "orange"],
            [4, 10, "orange"],
            [5, 11, "orange"],
            [6, 12, "orange"],
            [7, 0, "orange"],
            [8, 1, "orange"],
            [9, 2, "orange"],
            [10, 3, "orange"],
            [11, 4, "orange"],
            [12, 5, "orange"]
        ]
    },
    {
        "fileName": "9Add1.png",
        "title" : "9 Nodes, Cycle 1",
        "numPoints" : 9,
        "links" : [
            [0, 1, "red"],
            [1, 2, "red"],
            [2, 3, "red"],
            [3, 4, "red"],
            [4, 5, "red"],
            [5, 6, "red"],
            [6, 7, "red"],
            [7, 8, "red"],
            [8, 0, "red"],
        ]
    },
    {
        "fileName": "9Add2.png",
        "title" : "9 Nodes, Cycle 2",
        "numPoints" : 9,
        "links" : [
            [0, 2, "green"],
            [1, 3, "green"],
            [2, 4, "green"],
            [3, 5, "green"],
            [4, 6, "green"],
            [5, 7, "green"],
            [6, 8, "green"],
            [7, 0, "green"],
            [8, 1, "green"],
        ]
    },
    {
        "fileName": "9Add3.png",
        "title" : "9 Nodes, Cycle 3",
        "numPoints" : 9,
        "links" : [
            [0, 3, "blue"],
            [1, 4, "blue"],
            [2, 5, "blue"],
            [3, 6, "blue"],
            [4, 7, "blue"],
            [5, 8, "blue"],
            [6, 0, "blue"],
            [7, 1, "blue"],
            [8, 2, "blue"],
        ]
    },
    {
        "fileName": "9Add4.png",
        "title" : "9 Nodes, Cycle 4",
        "numPoints" : 9,
        "links" : [
            [0, 4, "yellow"],
            [1, 5, "yellow"],
            [2, 6, "yellow"],
            [3, 7, "yellow"],
            [4, 8, "yellow"],
            [5, 0, "yellow"],
            [6, 1, "yellow"],
            [7, 2, "yellow"],
            [8, 3, "yellow"],
        ]
    },
    {
        "fileName": "6Add1.png",
        "title" : "6 Nodes, Cycle 1",
        "numPoints" : 6,
        "links" : [
            [0, 1, "red"],
            [1, 2, "red"],
            [2, 3, "red"],
            [3, 4, "red"],
            [4, 5, "red"],
            [5, 0, "red"],
        ]
    },
    {
        "fileName": "6Add2.png",
        "title" : "6 Nodes, Cycle 2",
        "numPoints" : 6,
        "links" : [
            [0, 2, "green"],
            [1, 3, "green"],
            [2, 4, "green"],
            [3, 5, "green"],
            [4, 0, "green"],
            [5, 1, "green"],
        ]
    },
    {
        "fileName": "6Add3.png",
        "title" : "6 Nodes, Cycle 3",
        "numPoints" : 6,
        "links" : [
            [0, 3, "blue"],
            [1, 4, "blue"],
            [2, 5, "blue"],
            [3, 0, "blue"],
            [4, 1, "blue"],
            [5, 2, "blue"],
        ]
    },
]

font = ImageFont.truetype("arial.ttf", 150)

def TextPointLocation(index, count):
    angle = index / count * math.pi * 2.0
    angle = angle - math.pi / 2    
    return [math.cos(angle) * circleTextRadius + w / 2.0, math.sin(angle) * circleTextRadius + h / 2.0 + circleVerticalOffset]

def PointLocation(index, count):
    angle = index / count * math.pi * 2.0
    angle = angle - math.pi / 2    
    return [math.cos(angle) * circleRadius + w / 2.0, math.sin(angle) * circleRadius + h / 2.0 + circleVerticalOffset]

for settings in diagramSettingsList:
    img = Image.new("RGB", (w, h))
    img1 = ImageDraw.Draw(img)  

    img1.rectangle([0, 0, w, h], fill="white")

    img1.text([0, 0], settings["title"], font = font, align ="center", fill="black")

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
