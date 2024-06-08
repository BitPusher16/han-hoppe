import requests
import os
import cv2 as cv
import numpy as np


def download_and_tessellate(
    tile_url,
    sat_qk_nw,
    sat_w_tiles,
    sat_h_tiles,
    aerial_lod
):

    # sat tile:
    # https://t.ssl.ak.tiles.virtualearth.net/tiles/
    # a021230030223.jpeg?g=14543&n=z&prx=1
    # 021230030223 has 12 digits.

    #tile_url = 'https://t.ssl.ak.tiles.virtualearth.net/tiles/' + \
    #    'a[QUADKEY].jpeg?g=14543&n=z&prx=1'

    #sat_qk_nw = '021230021313'
    #sat_qk_nw = '21'
    #sat_w_tiles = 2
    #sat_h_tiles = 2

    # https://t.ssl.ak.tiles.virtualearth.net/tiles/
    # a02123003023130.jpeg?g=14543&n=z&prx=1
    #url = 'https://t.ssl.ak.tiles.virtualearth.net/tiles/' + \
    #    'a02123003023130.jpeg?g=14543&n=z&prx=1'

    sat_x_nw, sat_y_nw, sat_lod = qk_to_x_y_lod(sat_qk_nw)
    #print(sat_x_nw, sat_y_nw, sat_lod)

    for x in range(sat_x_nw, sat_x_nw + sat_w_tiles):
        for y in range(sat_y_nw, sat_y_nw + sat_h_tiles):
            sat_qk_curr = x_y_lod_to_qk(x, y, sat_lod)

            url = tile_url.replace('[QUADKEY]', sat_qk_curr)
            out_path = os.path.join(os.getcwd(),
                'data/rasters/tile_cache/' + sat_qk_curr + '.jpg')
            if os.path.exists(out_path):
                print('exists: ' + out_path)
            else:
                print('downloading: ' + out_path)
                response = requests.get(url)
                with open(out_path, 'wb') as out_file:
                    out_file.write(response.content)

    # tesselate tiles to create mosaic.
    mosaic = np.zeros((256 * sat_h_tiles,256 * sat_w_tiles,3), np.uint8)
    mosaic_out_path = os.path.join(os.getcwd(),
        'data/rasters/mosaics/021230021313_mos.jpg')

    for x in range(sat_x_nw, sat_x_nw + sat_w_tiles):
        for y in range(sat_y_nw, sat_y_nw + sat_h_tiles):
            sat_qk_curr = x_y_lod_to_qk(x, y, sat_lod)

            tile_path = os.path.join(os.getcwd(),
                'data/rasters/tile_cache/' + sat_qk_curr + '.jpg')
            img = cv.imread(tile_path)

            x_offset = x - sat_x_nw
            y_offset = y - sat_y_nw
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



