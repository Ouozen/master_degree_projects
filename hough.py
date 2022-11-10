import sys, os.path, cv2, numpy as np


def gradient_img(img: np.ndarray) -> np.ndarray:
    hor_grad = (img[1:, :] - img[:-1, :])[:, :-1]
    ver_grad = (img[:, 1:] - img[:, :-1])[:-1:, :]
    magnitude = np.sqrt(hor_grad ** 2 + ver_grad ** 2)
    return magnitude


def hough_transform(
        img: np.ndarray,
        n_rhos: int,
        n_thetas: int
): # -> (np.ndarray, np.ndarray, np.ndarray):
    accum = np.zeros((n_rhos, n_thetas))
    
    thetas = np.linspace(-np.pi / 2, np.pi / 2, n_thetas, endpoint=False)
    img_pos = np.array(np.nonzero(img))
    img_pos[0] = -img_pos[0] + (img.shape[0]) # чтобы ось шла сниху вверх, а не наоборот
    sin_theta = np.sin(thetas)
    cos_theta = np.cos(thetas)

    rho_temp = np.zeros((len(img_pos[0]), n_thetas))
    counter = 0
    for x, y in zip(img_pos[1], img_pos[0]):
        rho_temp[counter] = x * cos_theta + y * sin_theta
        counter += 1

    start_rho, end_rho = np.min(rho_temp), np.max(rho_temp)
    rhos = np.linspace(start_rho, end_rho, n_rhos)

    diff_rhos = rhos[1] - rhos[0]
    num_element = rho_temp - rhos[0]
    num_element = num_element // diff_rhos
    num_element = np.where(num_element % diff_rhos < diff_rhos / 2, num_element, num_element + 1)

    # соберем аккумулятор
    theta_list = [i for i in range(n_thetas)] * len(img_pos[0])
    for p, th in zip(num_element.flatten(), theta_list):
        accum[int(p)][th] += 1
    # перевернем аккумулятор потому что в numpy у нас отсчет сверху
    accum = np.flip(accum, 0)
    accum = accum.astype('uint32')

    return accum, thetas, rhos


def get_lines(
        ht_map: np.ndarray,
        n_lines: int,
        min_rho_line_diff: int,
        min_theta_line_diff: int,
) -> np.ndarray:
    result = np.zeros((n_lines, 2))
    nonzero = np.nonzero(ht_map)
    if not len(nonzero[0]):
        print('На изображении в пространстве Хафа отсутствуют прямые.')
        return []

    rho, theta = np.divmod(ht_map.argmax(), ht_map.shape[1])
    ht_map[rho, theta] = 0

    result[0] = [theta, rho]
    counter = 1

    while counter < n_lines: # counter - 1 чтобы не выйти за пределы len(result[:counter, :])

        if ht_map.max(): # если мы начинаем перебирать нули, то никаких прямых больше нет
            rho, theta = np.divmod(ht_map.argmax(), ht_map.shape[1])
            #rho = -rho + (ht_map[0])
            rho_line_diff = np.abs(result[:counter,1:2].flatten() - rho)
            rho_bool = np.sum(rho_line_diff < min_rho_line_diff)
            theta_line_diff = np.abs(result[:counter,0:1].flatten() - theta)
            theta_bool = np.sum(theta_line_diff < min_theta_line_diff)
            ht_map[rho, theta] = 0

            if rho_bool or theta_bool:
                pass
            else:
                result[counter] = [theta, rho]
                counter += 1
        else:
            break

    result[:, 1:2] = -result[:, 1:2] + ht_map.shape[0]
    result = result.astype('uint32')

    return  result


def main():
    assert len(sys.argv) == 9
    src_path, dst_ht_path, dst_lines_path, n_rhos, n_thetas, \
        n_lines, min_rho_line_diff, min_theta_line_diff = sys.argv[1:]

    n_rhos = int(n_rhos)
    assert n_rhos > 0

    n_thetas = int(n_thetas)
    assert n_thetas > 0

    n_lines = int(n_lines)
    assert n_lines > 0

    min_rho_line_diff = int(min_rho_line_diff)
    assert min_rho_line_diff > 0

    min_theta_line_diff = int(min_theta_line_diff)
    assert min_theta_line_diff > 0

    assert os.path.exists(src_path)
    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    assert img is not None

    gradient = gradient_img(img.astype(np.float32))
    ht_map, rhos, thetas = hough_transform(img, n_rhos, n_thetas)

    dst_ht_map = ht_map.astype(np.float32)
    dst_ht_map /= dst_ht_map.max() / 255
    dst_ht_map = dst_ht_map.round().astype(np.uint8)
    cv2.imwrite(dst_ht_path, dst_ht_map)

    lines = get_lines(ht_map, n_lines, min_rho_line_diff, min_theta_line_diff)
    with open(dst_lines_path, 'w') as fout:
        for rho_idx, theta_idx in lines:
            fout.write(f'{rhos[rho_idx]:.3f}, {thetas[theta_idx]:.3f}\n')


if __name__ == '__main__':
    main()
