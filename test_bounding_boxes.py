# Imports the Google Cloud client library
from google.cloud import vision
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from pathlib import Path
import os


def localize_objects(path):
    """Localize objects in the local image.

    Args:
    path: The path to the local file.
    """
    from google.cloud import vision

    client = vision.ImageAnnotatorClient()

    with open(path, "rb") as image_file:
        content = image_file.read()
    image = vision.Image(content=content)

    objects = client.object_localization(image=image).localized_object_annotations

    print(f"Number of objects found: {len(objects)}")
    for object_ in objects:
        print(f"\n{object_.name} (confidence: {object_.score})")
        print("Normalized bounding polygon vertices: ")
        for vertex in object_.bounding_poly.normalized_vertices:
            print(f" - ({vertex.x}, {vertex.y})")

    return objects


def plot_bboxes(
  image_file: str,
  bboxes: List[List[float]],
  xywh: bool = True,
  labels: Optional[List[str]] = None
) -> None:
    """
    Args:
      image_file: str specifying the image file path
      bboxes: list of bounding box annotations for all the detections
      xywh: bool, if True, the bounding box annotations are specified as
        [xmin, ymin, width, height]. If False the annotations are specified as
        [xmin, ymin, xmax, ymax]. If you are unsure what the mode is try both
        and check the saved image to see which setting gives the
        correct visualization.

    """
    fig = plt.figure()

    # add axes to the image
    ax = fig.add_axes([0, 0, 1, 1])

    image_folder = os.path.abspath(os.getcwd())

    # read and plot the image
    image = plt.imread(image_file)
    plt.imshow(image)

    img_arr = np.array(image)
    image_width = img_arr.shape[1]
    image_height = img_arr.shape[0]

    # Iterate over all the bounding boxes
    for i, bbox in enumerate(bboxes):
        if xywh:
          xmin, ymin, w, h = bbox
          xmin = xmin*image_width
          ymin = ymin*image_height
          w = w*image_width
          h = w*image_height
        else:
          xmin, ymin, xmax, ymax = bbox
          xmin = xmin*image_width
          xmax = xmax*image_width
          ymin = ymin*image_height
          ymax = ymax*image_height
          w = (xmax - xmin)
          h = (ymax - ymin)

        # add bounding boxes to the image
        box = patches.Rectangle(
            (xmin, ymin), w, h, edgecolor="red", facecolor="none"
        )

        ax.add_patch(box)

        if labels is not None:
          rx, ry = box.get_xy()
          cx = rx + box.get_width()/2.0
          cy = ry + box.get_height()/8.0
          l = ax.annotate(
            labels[i],
            (cx, cy),
            fontsize=8,
            fontweight="bold",
            color="white",
            ha='center',
            va='center'
          )
          l.set_bbox(
            dict(facecolor='red', alpha=0.5, edgecolor='red')
          )

    plt.axis('off')
    outfile = os.path.join(image_folder, "image_bbox.png")
    fig.savefig(outfile)

    print("Saved image with detections to %s" % outfile)


if __name__ == "__main__":
    img_source = 'data/nerf_data/nerf_llff_data/fern/images_4/image000.png'
    objects = localize_objects(img_source)

    # for obj in objects:
    #     #print(obj)
    #     vertices = obj.bounding_poly.normalized_vertices
    #     display(drawVertices(img_source, vertices, obj.name))

    bboxes = []
    labels = []
    for obj in objects:
        xs = set()
        ys = set()
        vertices = obj.bounding_poly.normalized_vertices
        for vert in vertices:
            xs.add(vert.x)
            ys.add(vert.y)
        min_x = min(xs)
        min_y = min(ys)
        max_x = max(xs)
        max_y = max(ys)
        vertices = [min_x, min_y, max_x, max_y]
        bboxes.append(vertices)
        labels.append(obj.name)

    print(bboxes)
    print(labels)

    plot_bboxes(img_source, bboxes, False, labels)
    
    print(objects)