import requests
import os
import cv2 as cv
import numpy as np


def download_and_tesselate(
    tile_url,
    sat_qk_nw,
    sat_w_tiles,
    sat_h_tiles,
    aerial_lod
):

    tile_cache = os.path.join(os.getcwd(), 'data/rasters/tile_cache')
    mosaic_dir = os.path.join(os.getcwd(), 'data/rasters/mosaics')

    get_tiles(sat_qk_nw, sat_w_tiles, sat_h_tiles, tile_url,
        tile_cache, '.jpg')

    tesselate(sat_qk_nw, sat_w_tiles, sat_h_tiles,
        tile_cache, '.jpg', mosaic_dir, '.png')

    get_tiles(sat_qk_nw + '00', sat_w_tiles*4, sat_h_tiles*4, tile_url,
        tile_cache, '.jpg')

    tesselate(sat_qk_nw + '00', sat_w_tiles*4, sat_h_tiles*4,
        tile_cache, '.jpg', mosaic_dir, '.png')


def get_tiles(qk_nw, w_tiles, h_tiles, tile_url, dest_dir, extension):
    x_nw, y_nw, lod = qk_to_x_y_lod(qk_nw)

    for x in range(x_nw, x_nw + w_tiles):
        for y in range(y_nw, y_nw + h_tiles):
            qk_curr = x_y_lod_to_qk(x, y, lod)

            url = tile_url.replace('[QUADKEY]', qk_curr)
            out_path = os.path.join(dest_dir, qk_curr + extension)
            if os.path.exists(out_path):
                print('tile exists: ' + out_path)
            else:
                print('downloading: ' + out_path)
                response = requests.get(url)
                with open(out_path, 'wb') as out_file:
                    out_file.write(response.content)

def tesselate(
    qk_nw, w_tiles, h_tiles, tile_cache, tile_ext,
    mosaic_dir, mosaic_ext
):
    x_nw, y_nw, lod = qk_to_x_y_lod(qk_nw)
    mosaic_out_path = os.path.join(mosaic_dir,
        qk_nw + '_w' + str(w_tiles) + '_h' + str(h_tiles) + mosaic_ext)

    if os.path.exists(mosaic_out_path):
        print('mosaic exists: ' + mosaic_out_path)
    else:
        print('tesselating: ' + mosaic_out_path)
        mosaic = np.zeros((256 * h_tiles,256 * w_tiles,3), np.uint8)

        for x in range(x_nw, x_nw + w_tiles):
            for y in range(y_nw, y_nw + h_tiles):
                qk_curr = x_y_lod_to_qk(x, y, lod)

                tile_path = os.path.join(tile_cache, qk_curr + tile_ext)
                img = cv.imread(tile_path)

                x_offset = x - x_nw
                y_offset = y - y_nw
                mosaic[y_offset*256:(y_offset+1)*256, x_offset*256:(x_offset+1)*256,:] = img
                #mosaic[0:256,0:256,:] = img

        cv.imwrite(mosaic_out_path, mosaic)

# https://learn.microsoft.com/en-us/bingmaps/articles/bing-maps-tile-system
def x_y_lod_to_qk(x, y, lod):
    qk = ''
    for i in range(lod, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if (x & mask) != 0:
            digit += 1
        if (y & mask) != 0:
            digit += 2
        qk += str(digit)
    return qk

def qk_to_x_y_lod(qk):
    lod = len(qk)
    tile_x = 0
    tile_y = 0
    for i in range(lod, 0, -1):
        #print(i)
        mask = 1 << (i-1)
        digit = qk[lod-i]
        if digit == '0':
            pass
        elif digit == '1':
            tile_x |= mask
        elif digit == '2':
            tile_y |= mask
        elif digit == '3':
            tile_x |= mask
            tile_y |= mask

    return (tile_x, tile_y, len(qk))



