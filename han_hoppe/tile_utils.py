import requests
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

debug_img_path = os.path.join(os.getcwd(), 'data/rasters/debug.png')

def download_and_tesselate(
    tile_url,
    sat_qk_nw,
    sat_w_tiles,
    sat_h_tiles,
    aerial_lod
):

    tile_cache = os.path.join(os.getcwd(), 'data/rasters/tile_cache')
    mosaic_dir = os.path.join(os.getcwd(), 'data/rasters/mosaics')
    #debug_img_path = os.path.join(mosaic_dir, 'debug.png')

    lod_diff = aerial_lod - len(sat_qk_nw)

    get_tiles(sat_qk_nw, sat_w_tiles, sat_h_tiles, tile_url,
        tile_cache, '.jpg')

    sat_img = tesselate(sat_qk_nw, sat_w_tiles, sat_h_tiles,
        tile_cache, '.jpg', mosaic_dir, '.png')

    get_tiles(sat_qk_nw + '0'*lod_diff, sat_w_tiles*(2**lod_diff), sat_h_tiles*(2**lod_diff),
      tile_url, tile_cache, '.jpg')

    aer_img = tesselate(sat_qk_nw + '0'*lod_diff, sat_w_tiles*(2**lod_diff), 
        sat_h_tiles*(2**lod_diff), tile_cache, '.jpg', mosaic_dir, '.png')

    # create pyramid, with sat in coarsest level (12 for bing maps)
    # and aerial in finest level (15).
    # populate intermediate levels with downscaled fine level.
    # upscale sat layer to size of aerial layer, and transfer structure from aerial.

    # populate pyramid from base to peak.
    # aerial image will have index 0.
    pyr = []
    #cv.imwrite(debug_img_path, aer_img)
    pyr.append(aer_img)
    aer_h, aer_w, *_ = aer_img.shape
    for i in range(1, aerial_lod - len(sat_qk_nw)):
        fac = 2 ** i
        intermed = cv.resize(
            aer_img, (aer_w//fac,aer_h//fac), interpolation=cv.INTER_AREA)
        pyr.append(intermed)
    pyr.append(sat_img)
    print('num layers: ' + str(len(pyr)))

    #sat_upscaled = cv.resize(sat_img, (aer_w,aer_h), interpolation=cv.INTER_LINEAR)
    ## shift sat upscaled 8px up and 8px to the left.
    ## bing aerial imagery is offset from satellite. shift to correct.
    #shift = 16
    #sat_upscaled[:aer_h-shift,:] = sat_upscaled[shift:,:]
    #sat_upscaled[:,:aer_w-shift] = sat_upscaled[:,shift:]
    #cv.imwrite(debug_img_path, sat_upscaled)


    # https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
    # opencv provides a function cv.filter2D() to convolve a kernel with an image.

    #structure = structure_transfer(sat_upscaled, aer_img)
    ##plt.imshow(structure[:,:,::-1])
    ##plt.show()
    #cv.imwrite(debug_img_path.replace('.png','_st.png'), structure)

    sat_h, sat_w, *_ = sat_img.shape
    structure = structure_transfer(
        np.copy(sat_img),
        cv.resize(aer_img, (sat_w,sat_h), interpolation=cv.INTER_AREA)
    )
    #plt.imshow(structure[:,:,::-1])
    #plt.show()

    for i in range(1, aerial_lod - len(sat_qk_nw)):
        # divide aerial image size by a factor to get target structure size.
        fac = 2 ** i

        num_layers = aerial_lod - len(sat_qk_nw) + 1
        structure_weight = i/(num_layers-1)
        #structure_weight = 0.0

        temp = np.copy(pyr[i]).astype(np.float32)
        structure_downsampled = cv.resize(
            structure, (aer_w//fac,aer_h//fac), interpolation=cv.INTER_AREA).astype(np.float32)

        # at index 0 (base of pyr), give all weight to original aerial image.
        # at index n-1 (peak of pyr), give all weight to structure image.
        temp = (temp * (1-structure_weight)) + (structure_downsampled * structure_weight)
        pyr[i] = temp.astype(np.uint8)

    #plt.imshow(pyr[1][:,:,::-1])
    #plt.show()

    # write out pyramid layers.
    for i in range(0, len(pyr)):
        cv.imwrite(
            os.path.join(os.getcwd(), 
                'data/rasters/mosaics/' + sat_qk_nw + '_lod' + str(aerial_lod-i) + '.png'),
            pyr[i]
        )



def structure_transfer(color, structure):
    print(color.shape)
    print(structure.shape)

    kernel_w= 21
    ker = (kernel_w, kernel_w)
    sig = (kernel_w - 1) // 5

    color_f = color.astype(np.float32)/255
    structure_f = structure.astype(np.float32)/255
    epsilon = np.zeros(color_f.shape, np.float32) + 0.000001

    color_lab = cv.cvtColor(color_f, cv.COLOR_BGR2Lab)
    color_mu = cv.GaussianBlur(color_lab, ker, sigmaX=sig, sigmaY=sig)
    color_diff = color_lab - color_mu
    color_sq = np.square(color_diff)
    color_sigma = np.sqrt(
        cv.GaussianBlur(color_sq, ker, sigmaX=sig, sigmaY=sig)
    )

    #color_mu_sq = np.square(color_mu)
    #color_sq = np.square(color_lab)
    #color_sq_mu = cv.GaussianBlur(color_sq, ker, sigmaX=sig, sigmaY=sig)
    #color_sigma = np.sqrt( color_sq_mu - color_mu_sq)

    #color_z = (color_lab - color_mu)/(color_sigma + epsilon)
    #plt.imshow(color_z[:,:,::-1])
    #plt.show()

    structure_lab = cv.cvtColor(structure_f, cv.COLOR_BGR2Lab)
    structure_mu = cv.GaussianBlur(structure_lab, ker, sigmaX=sig, sigmaY=sig)
    structure_diff = structure_lab - structure_mu
    structure_sq = np.square(structure_diff)
    structure_sigma = np.sqrt(
        cv.GaussianBlur( structure_sq, ker, sigmaX=sig, sigmaY=sig)
    )
    structure_z = (structure_lab - structure_mu)/(structure_sigma + epsilon)

    #debug_curr = np.divide((structure_lab - structure_mu),  (structure_sigma + epsilon))
    #debug_tmp = (debug_curr - debug_curr.min()) / (debug_curr.max() - debug_curr.min())

    #debug_tmp = structure_lab
    #plt.imshow(debug_tmp[:,:,:1:-1])
    #plt.show()

    #structure_mu_sq = np.square(structure_mu)
    #structure_sq = np.square(structure_lab)
    #structure_sq_mu = cv.GaussianBlur(structure_sq, ker, sigmaX=sig, sigmaY=sig)
    #structure_sigma = np.sqrt( structure_sq_mu - structure_mu_sq)
    #structure_z = (structure_lab - structure_mu)/(structure_sigma + epsilon)

    #plt.imshow(structure_z[:,:,:1:-1])
    #plt.show()

    v_prime = color_mu + np.multiply(structure_z, color_sigma)
    v_bgr = cv.cvtColor(v_prime, cv.COLOR_Lab2BGR)

    #deb = cv.normalize(structure_z, None, 255, 0, cv.NORM_MINMAX, cv.CV_8U)
    #cv.imwrite(debug_img_path, deb)
    #cv.imwrite(debug_img_path, cv.cvtColor((v_prime*255).astype(np.uint8), cv.COLOR_Lab2RGB))
    #color_mu_bgr = cv.cvtColor(color_mu, cv.COLOR_Lab2BGR)
    #color_mu_bgr = cv.cvtColor(color_mu, cv.COLOR_Lab2BGR)

    #cv.imwrite(debug_img_path, v_bgr*255)

    #plt.imshow(color_mu_bgr[:,:,::-1])
    #plt.show()

    return (v_bgr*255).astype(np.uint8)



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
        mosaic = cv.imread(mosaic_out_path)
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
    return mosaic

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



