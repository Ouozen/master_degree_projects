import sys, os.path, json
import numpy as np
from scipy.stats.distributions import chi2


def generate_data(
        img_size: tuple, line_params: tuple,
        n_points: int, sigma: float, inlier_ratio: float
) -> np.ndarray:
    # зададим точки на прямой
    shape_line = int(n_points * inlier_ratio)
    x = np.random.randint(0, img_size[0], shape_line)
    a, b, c = line_params
    y = (-a * x - c) / b # посчитаем значения y, если точка идеально лежит на прямой
    y = y.astype('float')
    y += np.sqrt(sigma) * np.random.standard_normal(shape_line) # добавим дисперсию

    # зададим выбросы
    x_outl = np.random.randint(0, img_size[0], n_points - shape_line)
    y_outl = np.random.randint(0, img_size[1], n_points - shape_line)
    
    x = np.concatenate((x, x_outl))
    y = np.concatenate((y, y_outl))
    data = np.stack((x, y), axis=-1)

    return data


def compute_ransac_threshold(
        alpha: float, sigma: float
) -> float:
    F = chi2.ppf(alpha, df=2)
    T = np.sqrt(F * sigma)

    return round(T, 3)


def compute_ransac_iter_count(
        conv_prob: float, inlier_ratio: float
) -> int:
    m = 2
    w = inlier_ratio
    N = np.log(1 - conv_prob) / np.log(1 - (w ** m))
    N = np.ceil(N)

    return int(N)


def compute_line_ransac(
        data: np.ndarray, threshold: float, iter_count: int
) -> tuple:
    count_inliers = 0
    best_sample = None

    for _ in range(iter_count):
        m = np.random.choice(np.arange(0, len(data)), size=(2,), replace=False)
        assert m[0] != m[1]
        points = np.take(data, m, axis=0)
        cross_prod = np.cross(points[1, :] - points[0, :], points[1, :] - data)
        support_length = np.linalg.norm(points[1, :] - points[0, :])
        distances = np.abs(cross_prod) / support_length
        num_inliers = np.sum(distances < threshold)

        if num_inliers > count_inliers:
            count_inliers = num_inliers
            best_sample = points
        
    if best_sample is not None:
        x1, y1 = best_sample[0]
        x2, y2 = best_sample[1]
        
        B = x1 - x2
        A = y2 - y1
        C = -x1 * y2 + x1 * y1 + y1 * x2 - y1 * x1

        A, B, C = A / A, round(B / A, 2), round(C / A, 2) 

        return A, B, C
    else:
        print('Прямая не найдена.')


def detect_line(params: dict) -> tuple:
    data = generate_data(
        (params['w'], params['h']),
        (params['a'], params['b'], params['c']),
        params['n_points'], params['sigma'], params['inlier_ratio']
    )
    threshold = compute_ransac_threshold(
        params['alpha'], params['sigma']
    )
    iter_count = compute_ransac_iter_count(
        params['conv_prob'], params['inlier_ratio']
    )
    detected_line = compute_line_ransac(data, threshold, iter_count)
    return detected_line


def main():
    assert len(sys.argv) == 2
    params_path = sys.argv[1]
    assert os.path.exists(params_path)
    with open(params_path) as fin:
        params = json.load(fin)
    assert params is not None

    """
    params:
    line_params: (a,b,c) - line params (ax+by+c=0)
    img_size: (w, h) - size of the image
    n_points: count of points to be used

    sigma - Gaussian noise
    alpha - probability of point is an inlier

    inlier_ratio - ratio of inliers in the data
    conv_prob - probability of convergence
    """

    detected_line = detect_line(params)
    print(detected_line)


if __name__ == '__main__':
    main()