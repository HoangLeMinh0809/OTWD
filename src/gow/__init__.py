import numpy as np
import ot

def polynomial(x, a, b):
    return a*x + b

def exponential(x, a, b, c):
    return a * np.exp(b*x + c)

def logarithm(x, a, b):
    return np.log(a*x + b)

def hyperbolic_tangent(x, a, b, c):
    return a * np.tanh(b*x + c)

def polynomial_with_degree(x, a, b, c):
    return a * pow(b*x, c)

def compute_f(function_info, x):
    match function_info[0]:
        case 'polynomial':
            return polynomial(x, function_info[1][0], function_info[1][1])
        case 'exponential':
            return exponential(x, function_info[1][0], function_info[1][1], function_info[1][2])
        case 'logarithm':
            return logarithm(x, function_info[1][0], function_info[1][1])
        case 'hyperbolic_tangent':
            return hyperbolic_tangent(x, function_info[1][0], function_info[1][1], function_info[1][2])
        case 'polynomial_with_degree':
            return polynomial_with_degree(x, function_info[1][0], function_info[1][1], function_info[1][2])

    return ValueError("Function not defined")

def compute_f_scale(function_info, x, i_scale, j_scale):
    '''Assume the input function is defined on the unit square.
    Scale the function according to the lengths of the two series.
    '''

    x_scaled = x / j_scale

    match function_info[0]:
        case 'polynomial':
            return i_scale * polynomial(x_scaled, function_info[1][0], function_info[1][1])
        case 'exponential':
            return i_scale * exponential(x_scaled, function_info[1][0], function_info[1][1], function_info[1][2])
        case 'logarithm':
            return i_scale * logarithm(x_scaled, function_info[1][0], function_info[1][1])
        case 'hyperbolic_tangent':
            return i_scale * hyperbolic_tangent(x_scaled, function_info[1][0], function_info[1][1], function_info[1][2])
        case 'polynomial_with_degree':
            return i_scale * polynomial_with_degree(x_scaled, function_info[1][0], function_info[1][1], function_info[1][2])
            
    return ValueError("Function not defined, valid function name:'polynomial', 'exponential', 'logarithm', 'hyperbolic_tangent', 'polynomial_with_degree'")

def compute_new_cost(old_D, alpha, F, LAMBDA3):
    '''
    Tính ma trận chi phí mới với vector trọng số mới
    (không scale)
    '''

    n = old_D.shape[0]
    m = old_D.shape[1]

    new_D = np.empty((n, m))

    for i in range(n):
        for j in range(m):
            new_D[i][j] = old_D[i][j] + LAMBDA3 * (i - float(np.dot(np.squeeze(np.asarray(alpha)), F[j])))**2 / (n**2)

    return new_D

def compute_new_cost2(old_D, w, F, LAMBDA1):
    '''Compute a new cost matrix using the new weight vector
    (takes series lengths into account).
    '''

    n = old_D.shape[0]
    m = old_D.shape[1]

    new_D = np.empty((n, m))

    for i in range(n):
        for j in range(m):
            new_D[i][j] = old_D[i][j] + LAMBDA1 * (i/n - float(np.dot(np.squeeze(np.asarray(w)), F[j]))/m)**2

    return new_D

def choose_initial_w(Y, V, num_function):
    '''Initialize the weight vector for the GOW objective.
    The vector has a single 1 and remaining entries equal to 0.
    '''

    initial_w = np.zeros((num_function, 1))
    min_S = np.Inf
    min_index = 0

    for i in range(num_function):
        sub = Y - V[:,[i]]
        sum_squared = np.sum(np.square(sub))

        if sum_squared < min_S:
            min_index = i
            min_S = sum_squared

    initial_w[min_index][0] = 1

    return initial_w

def gow_sinkhorn(a, b, D, function_list=[("polynomial", (1.0, 0)),], LAMBDA1=5, LAMBDA2=10, maxIter=15, epsilon=0.01, num_FW_iteration=100, show_details=False):
    '''Compute the GOW distance between two series.

    If the cost matrix D has shape n x m, the resulting transport matrix
    will also be n x m.

    Parameters
    ----------
    a : array-like, shape (dim_a,)
        Source histogram weights
    b : array-like, shape (dim_b,) or ndarray, shape (dim_b, n_hists)
        Target histogram weights
    D : array-like, shape (dim_a, dim_b)
        Loss/cost matrix
    function_list : list, optional
        Input functions used to control deformation paths
    LAMBDA1 : float, optional
        Regularization parameter for input functions
    LAMBDA2 : float, optional
        Regularization parameter for Sinkhorn
    maxIter: int, optional
        Maximum number of outer iterations (coordinate descent)
    epsilon: float, optional
        Stopping tolerance for weight updates
    num_FW_iteration: int, optional
        Number of Frank-Wolfe iterations
    show_details: bool, optional
        If True, return the GOW distance, transport matrix and weight vector

    Returns
    -------
    float
        GOW distance
    (float, array-like, array-like)
        GOW distance, transport matrix and weight vector (only if show_details==True)
    '''

    n = D.shape[0]
    m = D.shape[1]

    if len(a) == 0:
        a = np.full((n,), 1.0 / n)
    if len(b) == 0:
        b = np.full((m,), 1.0 / m)

    num_function = len(function_list)
    Y = np.empty((n*m, 1))
    V = np.empty((n*m, num_function))
    iterCount = 0
    F = np.empty((m, num_function))

    for j in range(m):
        for k in range(num_function):
            F[j][k] = compute_f(function_list[k], j)

    D_ = D

    while iterCount < maxIter:   
        iterCount = iterCount + 1

    # Optimize T
        T = ot.sinkhorn(a, b, D_, 1.0/LAMBDA2)

    # Optimize w
        index_Y = 0
        for i in range(n):
            for j in range(m):
                temp = np.sqrt(T[i][j])
                Y[index_Y][0] =  temp * i
                           
                for k in range(num_function):
                    V[index_Y][k] = temp * F[j][k]
                
                index_Y = index_Y + 1

        w_new = choose_initial_w(Y, V, num_function)
        
    # Frank-Wolfe loop
        for FW_index in range(num_FW_iteration):
            gradient = -2 * np.matmul(np.transpose(V), Y - np.matmul(V, w_new))
            min_index = np.argmin(gradient)
            s = np.zeros((num_function, 1))
            s[min_index][0] = 1
            FW_step_size = 2 / (FW_index + 2)
            w_new = w_new + FW_step_size*(s - w_new)

    # Check stopping condition
        if iterCount != 1:
            diff = (np.absolute(w_new - w_old)).max() 
            # diff = np.sqrt(np.sum((w_new - w_old) ** 2))

            if diff < epsilon:
                break

    # New cost matrix computed from the new w
        D_ = compute_new_cost2(D, w_new, F, LAMBDA1)
        w_old = w_new

    if show_details:
        return np.sum(D * T), T, w_new

    return np.sum(D * T)

def gow_sinkhorn_autoscale(a, b, D, function_list=[("polynomial", (1.0, 0)),], LAMBDA1=5, LAMBDA2=10, maxIter=15, epsilon=0.01, num_FW_iteration=100, show_details=False):
    '''Compute the GOW distance between two series with automatic scaling of
    the input functions according to the series lengths.

    Parameters
    ----------
    a : array-like, shape (dim_a,)
        Source histogram weights
    b : array-like, shape (dim_b,) or ndarray, shape (dim_b, n_hists)
        Target histogram weights
    D : array-like, shape (dim_a, dim_b)
        Loss/cost matrix
    function_list : list, optional
        Input functions used to control deformation paths
    LAMBDA1 : float, optional
        Regularization parameter for input functions
    LAMBDA2 : float, optional
        Regularization parameter for Sinkhorn
    maxIter: int, optional
        Maximum number of outer iterations (coordinate descent)
    epsilon: float, optional
        Stopping tolerance for weight updates
    num_FW_iteration: int, optional
        Number of Frank-Wolfe iterations
    show_details: bool, optional
        If True, return the GOW distance, transport matrix and weight vector

    Returns
    -------
    float
        GOW distance
    (float, array-like, array-like)
        GOW distance, transport matrix and weight vector (only if show_details==True)
    '''

    n = D.shape[0]
    m = D.shape[1]

    if len(a) == 0:
        a = np.full((n,), 1.0 / n)
    if len(b) == 0:
        b = np.full((m,), 1.0 / m)

    num_function = len(function_list)
    i_scale = n - 1
    j_scale = m - 1
    Y = np.empty((n*m, 1))
    V = np.empty((n*m, num_function))
    iterCount = 0
    F = np.empty((m, num_function))

    for j in range(m):
        for k in range(num_function):
            F[j][k] = compute_f_scale(function_list[k], j, i_scale, j_scale)

    w_0 = np.zeros(num_function)
    w_0[np.random.randint(num_function)] = 1
    D_ = compute_new_cost(D, w_0, F, LAMBDA1)

    while iterCount < maxIter:
        iterCount = iterCount + 1

    # Optimize T
        T = ot.sinkhorn(a, b, D_, 1.0/LAMBDA2)

    # Optimize w
        index_Y = 0
        for i in range(n):
            for j in range(m):
                temp = np.sqrt(T[i][j])
                Y[index_Y][0] =  temp * i
                           
                for k in range(num_function):
                    V[index_Y][k] = temp * F[j][k]
                
                index_Y = index_Y + 1

        w_new = choose_initial_w(Y, V, num_function)
        
    # Frank-Wolfe loop
        for FW_index in range(num_FW_iteration):
            gradient = -2 * np.matmul(np.transpose(V), Y - np.matmul(V, w_new))
            min_index = np.argmin(gradient)
            s = np.zeros((num_function, 1))
            s[min_index][0] = 1
            FW_step_size = 2 / (FW_index + 2)
            w_new = w_new + FW_step_size*(s - w_new)

    # New cost matrix computed from the new w
        D_ = compute_new_cost(D, w_new, F, LAMBDA1)

    # Check stopping condition
        if iterCount != 1:
            diff = (np.absolute(w_new - w_old)).max() 

            if diff < epsilon:
                break

        w_old = w_new

    if show_details:
        return np.sum(D * T), T, w_new

    return np.sum(D * T)

def gow_sinkhorn_autoscale_fixed(a, b, D, LAMBDA1=10, LAMBDA2=5, maxIter=15, epsilon=0.01, num_FW_iteration=100, show_details=False):
    '''
    Tính khoảng cách GOW giữa hai chuỗi.
    Nếu ma trận chi phí D có kích thước n x m,
    ma trận vận chuyển kết quả có kích thước n x m.
    Không cần các hàm đầu vào vì hàm này
    sử dụng 5 hàm đơn điệu cố định.

    Tham số
    ----------
    a : array-like, shape (dim_a,)
        Trọng số mẫu ở miền nguồn
    b : array-like, shape (dim_b,) hoặc ndarray, shape (dim_b, n_hists)
        Mẫu ở miền đích
    D : array-like, shape (dim_a, dim_b)
        Ma trận mất mát
    LAMBDA1 : float, tùy chọn
        Tham số điều chuẩn cho các hàm đầu vào
    LAMBDA2 : float, tùy chọn
        Tham số điều chuẩn cho Sinkhorn
    maxIter: int, tùy chọn
        Số vòng lặp tối đa cho vòng lặp chính (Coordinate Descent)
    epsilon: float, tùy chọn
        Ngưỡng dừng theo sai số
    num_FW_iteration: int, tùy chọn
        Số vòng lặp Frank-Wolfe
    show_details: bool, tùy chọn
        Trả về khoảng cách GOW, ma trận vận chuyển và vector trọng số nếu True

    Trả về
    -------
    float : khoảng cách GOW
    float, array-like, array-like:
        Khoảng cách GOW, ma trận vận chuyển và vector trọng số (chỉ trả về nếu show_details==True)
    '''

    n = D.shape[0]
    m = D.shape[1]
    i_scale = n - 1
    j_scale = m - 1

    if len(a) == 0:
        a = np.full((n,), 1.0 / n)
    if len(b) == 0:
        b = np.full((m,), 1.0 / m)

    # 5 hàm đơn điệu cố định
    func1 = ('polynomial_with_degree', (1.0, 1.0, 0.05))
    func2 = ('polynomial_with_degree', (1.0, 1.0, 0.28))
    func3 = ("polynomial", (1.0, 0)) 
    func4 = ('polynomial_with_degree', (1.0, 1.0, 3.2))
    func5 = ('polynomial_with_degree', (1.0, 1.0, 20))

    function_list = [func1, func2, func3, func4, func5]
    num_function = len(function_list)
    Y = np.empty((n*m, 1))
    V = np.empty((n*m, num_function))
    iterCount = 0
    F = np.empty((m, num_function))

    for j in range(m):
        for k in range(num_function):
            F[j][k] = compute_f_scale(function_list[k], j, i_scale, j_scale)

    w_0 = np.array([0, 0, 1, 0, 0])
    D_ = compute_new_cost(D, w_0, F, LAMBDA1)

    while iterCount < maxIter: 
        iterCount = iterCount + 1

        # Tối ưu T
        T = ot.sinkhorn(a, b, D_, 1.0/LAMBDA2)

        # Tối ưu w
        index_Y = 0
        for i in range(n):
            for j in range(m):
                temp = np.sqrt(T[i][j])
                Y[index_Y][0] =  temp * i
                           
                for k in range(num_function):
                    V[index_Y][k] = temp * F[j][k]
                
                index_Y = index_Y + 1

        w_new = choose_initial_w(Y, V, num_function)
        
        # Vòng lặp Frank-Wolfe
        for FW_index in range(num_FW_iteration):
            gradient = -2 * np.matmul(np.transpose(V), Y - np.matmul(V, w_new))
            min_index = np.argmin(gradient)
            s = np.zeros((num_function, 1))
            s[min_index][0] = 1
            FW_step_size = 2 / (FW_index + 2)
            w_new = w_new + FW_step_size*(s - w_new)

        # Ma trận chi phí mới từ w mới
        D_ = compute_new_cost(D, w_new, F, LAMBDA1)

        # Kiểm tra điều kiện dừng
        if iterCount != 1:
            diff = (np.absolute(w_new - w_old)).max() 

            if diff < epsilon:
                break

        w_old = w_new

    if show_details:
        return np.sum(D * T), T, w_new

    return np.sum(D * T)
