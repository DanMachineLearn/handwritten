# -*- encoding: utf-8 -*-
'''
@File		:	normalization_nonlinear_global.py
@Time		:	2024/05/13 10:09:24
@Author		:	dan
@Description:	根据 https://blog.csdn.net/rushkid02/article/details/9242415 的代码，实现了另一个图像整体非线性均衡归一化
'''
import numpy as np
import math

def forward_push_val(dst, dst_wid, dst_hei, val, x, y, scalex, scaley):
    fl = x - scalex / 2
    fr = x + scalex / 2
    ft = y - scaley / 2
    fb = y + scaley / 2

    l = int(fl)
    r = int(fr) + 1
    t = int(ft)
    b = int(fb) + 1

    l = max(min(l, dst_wid - 1), 0)
    r = max(min(r, dst_wid - 1), 0)
    t = max(min(t, dst_hei - 1), 0)
    b = max(min(b, dst_hei - 1), 0)

    for j in range(t, b + 1):
        for i in range(l, r + 1):
            xg = min(i + 1, fr) - max(i, fl)
            yg = min(j + 1, fb) - max(j, ft)

            if xg > 0 and yg > 0:
                dst[j, i] += int(xg * yg * val)

def forward_push_val2(dst, dst_wid, dst_hei, val, fl, ft, fr, fb, scalex, scaley):
    l = int(fl)
    r = int(fr) + 1
    t = int(ft)
    b = int(fb) + 1

    l = max(min(l, dst_wid - 1), 0)
    r = max(min(r, dst_wid - 1), 0)
    t = max(min(t, dst_hei - 1), 0)
    b = max(min(b, dst_hei - 1), 0)

    for j in range(t, b + 1):
        for i in range(l, r + 1):
            xg = min(i + 1, fr) - max(i, fl)
            yg = min(j + 1, fb) - max(j, ft)

            if xg > 0 and yg > 0:
                dst[j, i] += int(xg * yg * val)


RADIOFUNC_FIXED = 0
RADIOFUNC_ASPECT = 1
RADIOFUNC_SQUARE = 2
RADIOFUNC_CUBIC = 3
RADIOFUNC_SINE = 4

def aspect_radio_mapping(r1, dst_wid, dst_hei, ratio_preserve_func):
    if ratio_preserve_func == RADIOFUNC_ASPECT:
        return r1
    elif ratio_preserve_func == RADIOFUNC_SQUARE:
        return math.sqrt(r1)
    elif ratio_preserve_func == RADIOFUNC_CUBIC:
        return math.pow(r1, 1/3)
    elif ratio_preserve_func == RADIOFUNC_SINE:
        return math.sqrt(math.sin(math.pi * r1 / 2))
    else:
        return min(dst_wid, dst_hei) / max(dst_wid, dst_hei)

def remap(image, target_size : tuple[2] | list[2] = (64, 64), ratio_preserve_func = RADIOFUNC_ASPECT):
    '''
    src, 原图像
    src_wid, 图像宽度
    src_hei, 图像高度
    src_widstep, 图像宽度位数，
    region, 真实图像所在位置
    dst, 
    dst_wid, 
    dst_hei, 
    dst_widstep, 
    ratio_preserve_func = RADIOFUNC_ASPECT 运算算法
    '''
    src_wid = image.shape[1]
    src_hei = image.shape[0]
    dst = np.zeros(target_size, dtype=np.uint8)
    dst_wid = image.shape[1]
    dst_hei = image.shape[0]

    m00, m10, m01 = 0, 0, 0
    u20, u02 = 0, 0

    constval = 0.001


    for y in range(0, src_hei):
        for x in range(0, src_wid):
            m00 += image[y, x]
            m10 += x * image[y, x]
            m01 += y * image[y, x]

    if m00 == 0:
        return

    xc = m10 // m00
    yc = m01 // m00

    for y in range(0, src_hei):
        for x in range(0, src_wid):
            u20 += (x - xc) ** 2 * image[y, x]
            u02 += (y - yc) ** 2 * image[y, x]

    w1 = int(4 * math.sqrt(u20 / m00))
    h1 = int(4 * math.sqrt(u02 / m00))

    l = max(min(xc - w1 // 2, src_wid), 0)
    r = max(min(xc + w1 // 2 + 1, src_wid), 0)
    t = max(min(yc - h1 // 2, src_hei), 0)
    b = max(min(yc + h1 // 2 + 1, src_hei), 0)

    dx = np.zeros((b - t, r - l), dtype=float)
    dy = np.zeros((b - t, r - l), dtype=float)
    px = np.zeros((r - l,), dtype=float)
    py = np.zeros((b - t,), dtype=float)
    hx = np.zeros((r - l,), dtype=float)
    hy = np.zeros((b - t,), dtype=float)

    for y in range(t, b):
        run_start = -1
        run_end = -1
        for x in range(l, r):
            if image[y, x] < 128:
                if run_start < 0:
                    run_start = x
                    run_end = x
                else:
                    run_end = x
            else:
                if run_start < 0:
                    dx[y - t, x - l] = constval
                else:
                    d = 1.0 / (w1 + run_end - run_start + 1)
                    dx[y - t, x - l] = constval
                    for i in range(run_start, run_end + 1):
                        dx[y - t, i - l] = d
                    run_end = run_start = -1
        if run_start > 0:
            d = 1.0 / (w1 + run_end - run_start + 1)
            for i in range(run_start, run_end + 1):
                dx[y - t, i - l] = d

    for x in range(l, r):
        run_start = -1
        run_end = -1
        for y in range(t, b):
            if image[y, x] < 128:
                if run_start < 0:
                    run_start = y
                    run_end = y
                else:
                    run_end = y
            else:
                if run_start < 0:
                    dy[y - t, x - l] = constval
                else:
                    d = 1.0 / (h1 + run_end - run_start + 1)
                    dy[y - t, x - l] = constval
                    for i in range(run_start, run_end + 1):
                        dy[i - t, x - l] = d
                    run_end = run_start = -1
        if run_start > 0:
            d = 1.0 / (h1 + run_end - run_start + 1)
            for i in range(run_start, run_end + 1):
                dy[i - t, x - l] = d

    dx_sum = np.sum(dx)
    dy_sum = np.sum(dy)

    for y in range(t, b):
        py[y - t] = np.sum(dy[y - t, :]) / dy_sum

    for x in range(l, r):
        px[x - l] = np.sum(dx[:, x - l]) / dx_sum

    for x in range(l, r):
        hx[x - l] = np.sum(px[:x - l])

    for y in range(t, b):
        hy[y - t] = np.sum(py[:y - t])

    r1 = min(r - l, b - t) / max(r - l, b - t)
    r2 = aspect_radio_mapping(r1, dst_wid, dst_hei, ratio_preserve_func)
    dst.fill(0)

    if w1 > h1:
        w2 = dst_wid
        h2 = int(w2 * r2)
        xoffset = 0
        yoffset = (dst_hei - h2) / 2
    else:
        h2 = dst_hei
        w2 = int(h2 * r2)
        xoffset = (dst_wid - w2) / 2
        yoffset = 0

    xscale = w2 / w1
    yscale = h2 / h1

    for y in range(t, b):
        for x in range(l, r):
            x1 = w2 * hx[x - l]
            y1 = h2 * hy[y - t]

            if image[y, x] > 128:
                image[y, x] = image[y, x]

            if y == b - 1 or x == r - 1:
                forward_push_val(dst, dst_wid, dst_hei, image[y, x], x1 + xoffset, y1 + yoffset, xscale, yscale)
            else:
                x2 = w2 * hx[x - l + 1]
                y2 = h2 * hy[y - t + 1]
                forward_push_val2(dst, dst_wid, dst_hei, image[y, x], x1 + xoffset, y1 + yoffset, x2 + xoffset, y2 + yoffset, xscale, yscale)

    return dst


def main():
    '''
    
    '''
    
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.widgets import Slider
    # 支持中文
    plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号


    # 读取原始图像
    image_root = 'images/yi'
    image_paths = [f'{image_root}/1.jpg', 
                   f'{image_root}/2.jpg', 
                   f'{image_root}/3.jpg', 
                   f'{image_root}/4.jpg', 
                   f'{image_root}/5.jpg',
                   f'{image_root}/6.jpg']
    
    images = []
    for p in image_paths:
        images.append(cv2.imread(p, cv2.IMREAD_GRAYSCALE))
    
    fig, ax = plt.subplots(len(image_paths), 2)

    for i in range(len(image_paths)):
        ax_map : Axes = ax[i, 0]
        ax_map.imshow(images[i], cmap='gray')
        ax_map.set_title("原图像")

    def update_img():
        for i in range(len(image_paths)):
            ax_map : Axes = ax[i, 1]
            ax_map.clear()
            image_remap = remap(image=images[i])
            ax_map.imshow(image_remap, cmap='gray')
            ax_map.set_title("均衡化之后的图像")

    update_img()
    plt.show()
    pass

if __name__ == '__main__':
    main()
