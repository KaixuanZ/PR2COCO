import json
import argparse
import os
import cv2

def RectFN2ImgFN(fname):
    # transform the rectangle json filename to original image filename
    return '_'.join(fname.split('.')[0].split('_')[:2]) + '.png'

class Annotation:
    def __init__(self, image_id, rect, category_id, id):
        self.image_id = image_id
        self.iscrowd = 1                # for our data it's crowded
        self.category_id = category_id  # category id
        self.id = id                    # annotation id
        self.area = rect[1][0]*rect[1][1]
        self.segmentation = cv2.boxPoints(tuple(rect))  # a matrix of [[x,y],...]
        x_min, x_max, y_min, y_max = min(self.segmentation[:,0]), max(self.segmentation[:,0]), min(self.segmentation[:,1]), max(self.segmentation[:,1])
        self.bbox = [x_min, y_min, x_max-x_min, y_max-y_min]
        self.segmentation = self.segmentation.ravel().tolist()

    def to_dict(self):
        pass

class COCO:
    def __init__(self):
        self.annotations = []

    def addAnnotation(self, rect_fname, rect, category_id, id):
        annotation = Annotation(RectFN2ImgFN(rect_fname), rect, category_id, id)
        self.annotations.append(annotation)

    def to_Json(self, path):
        pass

def PR2COCO(ROI_index, coco, opt):
    # read in ROI_rect
    with open(os.path.join(opt.rect_path, opt.ROI, ROI_index)) as file:
        ROI_rect = json.load(file)

    # read in col_rects
    with open(os.path.join(opt.rect_path, opt.col, ROI_index)) as file:
        col_rects = json.load(file) # a list of rects [col_rects]

    # read in row_rects
    with open(os.path.join(opt.rect_path, opt.row, ROI_index)) as file:
        row_rects = json.load(file) # a dict of list if rects {col_i:[row_rects]}

    # read in classification results
    with open(os.path.join(opt.cls_path, ROI_index)) as file:
        cls = json.load(file)
    cls = cls['id']

    # transform them into COCO json format
    # use the first annotation:
    # Has a segmentation list of vertices (x, y pixel positions)
    # Has an area of 702 pixels (pretty small) and a bounding box of [473.07,395.93,38.65,28.67]
    # Is not a crowd (meaning it’s a single object)
    # Is category id of 18 (which is a dog)
    # Corresponds with an image with id 289343 (which is a person on a strange bicycle and a tiny dog)

    # add ROI_rect to coco
    coco.addAnnotation(ROI_index, ROI_rect, 0, 0)

    # add col_rect to coco
    for col_rect in col_rects:
        coco.addAnnotation(ROI_index, col_rect, 0, 1)

    #add section_rect to coco

    # add row_rect to coco
    i = 0
    for col in row_rects:
        for row_rect in row_rects[col]:
            coco.addAnnotation(ROI_index, row_rect, 0, 3)
            coco.addAnnotation(ROI_index, row_rect, 1, cls[i])
            i += 1

    import pdb;pdb.set_trace()

if __name__ == "__main__":
    # by default we use firm section
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, default="../raw_data/personnel-records/1954/scans/firm",help="path of where the img is saved")
    parser.add_argument("--cls_path", type=str, default="../results/personnel-records/1954/cls/CRF/firm",help="path of where the cls is saved")
    parser.add_argument("--rect_path", type=str, default="../results/personnel-records/1954/seg/firm", help="root path of where the rect is saved")
    parser.add_argument("--ROI", type=str, default="ROI_rect", help="rectangle of detected region of interest")
    parser.add_argument("--col", type=str, default="col_rect",help="rectangle of detected column")
    parser.add_argument("--row", type=str, default="row_rect", help="rectangle of detected row")
    parser.add_argument("--output_path", type=str, default="tmp.json", help="output path")

    opt = parser.parse_args()
    print(opt)

    clean_names = lambda x: [i for i in x if i[0] != '.']
    ROIs = os.listdir(os.path.join(opt.rect_path,opt.ROI))
    ROIs = sorted(clean_names(ROIs))

    coco = COCO()

    PR2COCO(ROIs[0], coco, opt)   #only process first ROI in this demo

    coco.to_Json(opt.output_path)
