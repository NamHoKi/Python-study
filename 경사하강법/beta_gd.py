import numpy as np

X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.dot(X, np.array([1, 2])) + 3

beta_gd = [10.1, 15.1, -6.5] # [1, 2, 3] 이 정답
X_ = np.array([np.append(x, [1]) for x in X]) # intercept 항 추가

for t in range(5000):
    error = y - X_ @ beta_gd
    # error = error / np.linalg.onrm(error)
    grad = - np.transpose(X_) @ error
    beta_gd = beta_gd - 0.01 * grad
  
print(beta_gd)


'''
학습률을 너무 작게 잡으면 수렴을 너무 늦게함
학습률을 너무 크게 잡으면 불안정한 움직임의 경사하강법
학습횟수가 너무 적으면 수렴이 잘 안됨
==> 학습률, 학습횟수를 적절하게
'''
