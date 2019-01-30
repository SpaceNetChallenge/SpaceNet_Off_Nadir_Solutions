import numpy as np
import util
import glob
import tifffile
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import cascaded_union
from shapely.wkt import loads, dumps
from skimage import measure
from patchify import PatchGenerator
from tqdm import tqdm
from pycocotools import mask as maskUtils
from models import utils
import json
import imgaug
from imgaug import augmenters as iaa


IMSIZE = (900,900)
p300 = PatchGenerator(stepsize=300, imsize=IMSIZE)
p512 = PatchGenerator(stepsize=512, imsize=IMSIZE)


#used to filter out predictions in the ovelap region autogenerate
def get_overlap_poly():
    mask = np.zeros((900, 900))
    mask[388:512, 0:900] = 1
    my, _, _ = util.mask_to_multipolygons(mask)
    mask = np.zeros((900, 900))
    mask[0:900, 388:512] = 1
    mx, _, _ = util.mask_to_multipolygons(mask)
    return mx, my


def get_mpan_image_patches(ms,pan,patch_creator):
    ms,pan =  tifffile.imread(ms), tifffile.imread(pan)
    is_blank = np.sum(pan)==0
    if is_blank:
        return None, None
    if np.argmin(ms.shape) == 2:
        ms = np.transpose(ms, (2, 0, 1))
    img = util.pansharpen(ms, pan)
    img_patches, _, _ = patch_creator.create(img=img)
    return img_patches, img


def get_rgb_image_patches(rgb, patch_creator):
    img = tifffile.imread(rgb)
    is_blank = np.sum(np.sum(img))==0
    if is_blank:
        return None,None
    if np.argmin(img.shape) == 0:
        img = np.transpose(img, (1, 2, 0))
    #img = util.stretch_8bit(img[:, :, [2, 1, 0]],lower_percent=2,higher_percent=98)
    img = img[:,:,[3,2,1,0]]
    img_patches, _ ,_ = patch_creator.create(img=img)
    return img_patches,img


def fix_multipolygon(mpoly):
    # try to fix it by buffering by 1 as close by contours are most likely to be of same object
    poly = []
    for i, item in enumerate(mpoly):
        p = item.buffer(1)   #maybe 0
        if p.is_valid:
            poly.append(p)
    poly = cascaded_union(poly).buffer(0)
    if not poly.geom_type == 'Polygon':
        poly = poly.convex_hull
    return poly


def fix_poly(polys):
    poly = []
    for p in polys:
        p = p.buffer(0)
        if p.is_valid:
            poly.append(p)
    # may return empty poly check downstream
    poly = cascaded_union(poly).buffer(0)
    return poly


#polygon need to be shifted as its part of larger tile
def polygonize_and_shift(poly,shifts):
    sx,sy = shifts
    def shift_poly(poly, shift):
        sx, sy = shift
        exterior_coords = poly.exterior.coords[:]
        interior_coords = []
        for interior in poly.interiors:
            interior_coords += interior.coords[:]
        new_exterior = []
        for x, y in exterior_coords:
            new_exterior.append([x + sx, y + sy])
        new_interior = []
        for x, y in interior_coords:
            new_interior.append([x + sx, y + sy])
        if new_interior != []:
            poly = Polygon(shell=new_exterior, holes=[new_interior]).buffer(0)
        else:
            poly = Polygon(shell=new_exterior).buffer(0)
        return poly

    if poly != []:
        poly = fix_poly(poly)
        if not poly.is_empty:
            if poly.geom_type == 'MultiPolygon':
                poly = fix_multipolygon(poly)
            polys = shift_poly(poly, (sx, sy))
            if (polys.geom_type!=poly.geom_type):
                polys=polys.convex_hull
            return  polys
        else:
            return None
    else:
        return None


def close_contour(contour):
    if not np.array_equal(contour[0],contour[-1]):
        contour = np.vstack((contour,contour[0]))
    return contour


def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation
    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        segmentation = [np.clip(i,0.0,i).tolist() for i in segmentation]
        polygons.append(segmentation)

    return polygons


def get_predictions_spacenet(image_id, img_patches, model, shifts, conf_thres=0.95):
    annotations = []
    bid = 0
    for i, image in enumerate(img_patches):
        sx, sy = shifts[i]
        prediction = model.detect([image], verbose=0)
        r = prediction[0]
        if r is not None:
            for idx, class_id in enumerate(r["class_ids"]):
                if class_id == 1:
                    confidence = r["scores"][idx]
                    if confidence>=conf_thres:
                        mask = r["masks"][:, :, idx].astype(np.uint8)
                        bbox = np.around(r["rois"][idx], 1)
                        bbox = [float(x) for x in bbox]
                        try:
                             #polies = util.mask_to_polygons(mask,epsilon=0.0, min_area=10)
                             seg = binary_mask_to_polygon(mask)
                             polies = MultiPolygon([Polygon(item) for item in seg])
                             if polies.geom_type=='Polygon':
                                polies = [polies]
                             poly = polygonize_and_shift(polies, (sx, sy))
                             if poly is not None:
                                bid += 1
                                result = [image_id, bid, poly, confidence, bbox]
                                annotations.append(result)
                        except:
                             print('Got polygon creation error', image_id, bid)  #log it

    if annotations == []:
        annotations = [[image_id, -1, 'POLYGON EMPTY', 'POLYGON EMPTY',[]]]
    return annotations


def get_final_annotations(image_id,anns,mx,my):
    keep = []
    discard = []
    building_id = 0
    for item in anns[0]:
        image_id, bid, poly, _, _ = item
        if bid == -1:
            continue
        if poly.intersects(mx) or poly.intersects(my):
            discard.append([item[0], 0, dumps(poly,3), item[3]])
        else:
            keep.append([item[0], building_id, dumps(poly, 3), item[3]])
            building_id += 1

    apoly1 = []
    for item in anns[1]:
        image_id, bid, poly, _, _ = item
        if bid == -1:
            continue
        if (poly.intersects(my)):
            apoly1.append(poly)

    apoly1 = cascaded_union(MultiPolygon(apoly1)).buffer(0)
    if not apoly1.is_empty:
        if apoly1.geom_type == 'Polygon':
            apoly11 = [apoly1]
        else:
            apoly11 = apoly1
        for ap1 in apoly11:
            keep.append([image_id, building_id, dumps(ap1, 3), 1])
            building_id += 1

    apoly2 = []
    for item in anns[2]:
        image_id, bid, poly, _ ,_ = item
        if bid == -1:
            continue
        if (poly.intersects(mx) and not poly.intersects(apoly1)):
            apoly2.append(poly)
    apoly2 = cascaded_union(MultiPolygon(apoly2)).buffer(0)
    if not apoly2.is_empty:
        if apoly2.geom_type == 'Polygon':
            apoly2 = [apoly2]
        for ap2 in apoly2:
            keep.append([image_id, building_id, dumps(ap2, 3), 1])
            building_id += 1

    return keep


def predict_mpan512(subdir, model,conf_thres=0.95):
    mx, my = get_overlap_poly()
    patch_creator = p512
    keep_all = []
    tifimages_pan = glob.glob(''.join([subdir, 'PAN/', '*.tif']))
    tifimages_ms = [''.join([subdir, 'MS/', item.split('/')[-1].replace('PAN', 'MS')]) for item in tifimages_pan]
    for ms, pan in list(zip(tifimages_ms, tifimages_pan)):
        anns = {}
        image_id = ms.split('/')[-1].replace('.tif', '').replace('MS_', '')
        img_patches,img = get_mpan_image_patches(ms,pan,patch_creator)
        if img is None:
            keep_all.extend([[image_id,-1,'POLYGON EMPTY','POLYGON EMPTY']])
            continue
        shifts = {}
        for k in patch_creator.coords.keys():
            shifts[k] = (patch_creator.coords[k][2], patch_creator.coords[k][0])
        spacenet_predctions_default = get_predictions_spacenet(image_id, img_patches, model, shifts,conf_thres=conf_thres)
        anns[0] = spacenet_predctions_default
        xx = [[0, 512, 256, 768],[388,900,256,768]]
        yy = [[256, 768, 0, 512],[256,768,388,900]]
        shift_x = {0: (0, 256), 1: (388, 256)}
        shift_y = {k: (v[1], v[0]) for k, v in shift_x.items()}
        img_patches = []
        for ix in yy:
            img_patches.append(img[ix[0]:ix[1], ix[2]:ix[3], :])
        img_patches = np.asarray(img_patches)
        spacenet_predctions_x = get_predictions_spacenet(image_id, img_patches, model, shift_x,conf_thres=conf_thres)
        anns[1] = spacenet_predctions_x
        img_patches = []
        for ix in xx:
            img_patches.append(img[ix[0]:ix[1], ix[2]:ix[3], :])
        img_patches = np.asarray(img_patches)
        spacenet_predctions_y = get_predictions_spacenet(image_id, img_patches, model, shift_y,conf_thres=conf_thres)
        anns[2] = spacenet_predctions_y
        keep = get_final_annotations(image_id,anns,mx,my)
        if keep == []:
            keep = [[image_id, -1, 'POLYGON EMPTY', "POLYGON EMPTY"]]
        keep_all.extend(keep)
    return keep_all


def predict_rgb512(subdir, model,conf_thres=0.95):
    mx, my = get_overlap_poly()
    patch_creator = p512
    keep_all = []
    tifimages = glob.glob(''.join([subdir, 'Pan-Sharpen/', '*.tif']))
    for tifimage in tifimages:
        anns = {}
        image_id = tifimage.split('/')[-1].replace('.tif', '').replace('Pan-Sharpen_', '')
        img_patches, img = get_rgb_image_patches(tifimage,patch_creator)
        if img is None:
            keep_all.extend([[image_id,-1,'POLYGON EMPTY','POLYGON EMPTY']])
            continue

        shifts = {}
        for k in patch_creator.coords.keys():
            shifts[k] = (patch_creator.coords[k][2], patch_creator.coords[k][0])
        spacenet_predctions_default = get_predictions_spacenet(image_id, img_patches, model, shifts,conf_thres=conf_thres)
        anns[0] = spacenet_predctions_default
        xx = [[0, 512, 256, 768],[388,900,256,768]]
        yy = [[256, 768, 0, 512],[256,768,388,900]]
        shift_x = {0: (0, 256), 1: (388, 256)}
        shift_y = {k: (v[1], v[0]) for k, v in shift_x.items()}
        img_patches = []
        for ix in yy:
            img_patches.append(img[ix[0]:ix[1], ix[2]:ix[3], :])
        img_patches = np.asarray(img_patches)
        spacenet_predctions_x = get_predictions_spacenet(image_id, img_patches, model, shift_x,conf_thres=conf_thres)
        anns[1] = spacenet_predctions_x
        img_patches = []
        for ix in xx:
            img_patches.append(img[ix[0]:ix[1], ix[2]:ix[3], :])
        img_patches = np.asarray(img_patches)
        spacenet_predctions_y = get_predictions_spacenet(image_id, img_patches, model, shift_y,conf_thres=conf_thres)
        anns[2] = spacenet_predctions_y
        keep = get_final_annotations(image_id, anns, mx, my)
        if keep==[]:
            keep=[[image_id,-1,'POLYGON EMPTY',"POLYGON EMPTY"]]
        keep_all.extend(keep)
    return keep_all


def get_predictions_coco(dataset, model, subname='tmp.json',conf_thres=0.0):
    final_predictions = []
    for i in tqdm(dataset.image_ids):
        im = dataset.load_image(i)
        images = [im]
        image_id = dataset.imgIds[i].replace('.jpg','').replace('.tif','')
        predictions = model.detect(images, verbose=0)
        r = predictions[0]
        for idx, class_id in enumerate(r["class_ids"]):
            if class_id == 1:
                confidence = r["scores"][idx]
                if confidence >= conf_thres:
                    mask = r["masks"].astype(np.uint8)[:, :, idx]
                    bbox = np.around(r["rois"][idx], 1)
                    bbox = [float(x) for x in bbox]
                    result = {}
                    result["image_id"] = image_id
                    result["category_id"] = 100
                    result["score"] = float(r["scores"][idx])
                    mask = maskUtils.encode(np.asfortranarray(mask))
                    mask["counts"] = mask["counts"].decode("UTF-8")
                    result["segmentation"] = mask
                    result["bbox"] = [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]]
                    final_predictions.append(result)
    fp = open(subname, "w")
    fp.write(json.dumps(final_predictions))
    fp.close()


