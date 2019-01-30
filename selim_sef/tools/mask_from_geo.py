import os

import click
from spacenetutilities.labeltools import coreLabelTools as cLT

def masks_from_geojsons(geojson_dir, im_src_dir, mask_dest_dir,
                        skip_existing=False, verbose=False):
    """Create mask images from geojsons.

    Arguments:
    ----------
    geojson_dir (str): Path to the directory containing geojsons.
    im_src_dir (str): Path to a directory containing geotiffs corresponding to
        each geojson. Because the georegistration information is identical
        across collects taken at different nadir angles, this can point to
        geotiffs from any collect, as long as one is present for each geojson.
    mask_dest_dir (str): Path to the destination directory.

    Creates a set of binary image tiff masks corresponding to each geojson
    within `mask_dest_dir`, required for creating the training dataset.

    """
    if not os.path.exists(geojson_dir):
        raise NotADirectoryError(
            "The directory {} does not exist".format(geojson_dir))
    if not os.path.exists(im_src_dir):
        raise NotADirectoryError(
            "The directory {} does not exist".format(im_src_dir))
    geojsons = [f for f in os.listdir(geojson_dir) if f.endswith('json')]
    ims = [f for f in os.listdir(im_src_dir) if f.endswith('.tif')]
    for geojson in geojsons:
        chip_id = os.path.splitext('_'.join(geojson.split('_')[1:]))[0]
        dest_path = os.path.join(mask_dest_dir, 'mask_' + chip_id + '.tif')
        if os.path.exists(dest_path) and skip_existing:
            if verbose:
                print('{} already exists, skipping...'.format(dest_path))
            continue
        matching_im = [i for i in ims if chip_id in i][0]
        # assign output below so it's silent
        g = cLT.createRasterFromGeoJson(os.path.join(geojson_dir, geojson),
                                        os.path.join(im_src_dir, matching_im),
                                        dest_path)


@click.command()
@click.option('--train-dir', type=str, help='path to dataset directory (it should include train dir)')
def main(train_dir):
    os.makedirs(os.path.join("train_labels", "masks"), exist_ok=True)
    masks_from_geojsons(os.path.join(train_dir, "geojson/spacenet-buildings"),
                        os.path.join(train_dir, "Atlanta_nadir10_catid_1030010003993E00", "Pan-Sharpen"),
                        os.path.join("train_labels", "masks"))
if __name__ == '__main__':
    main()