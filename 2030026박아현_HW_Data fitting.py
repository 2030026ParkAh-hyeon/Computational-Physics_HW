import os
os.chdir(os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid

# 데이터 읽어들이기
bins, count = [], []
with open("hist2.csv", "r") as f:
    for line in f.readlines():
        _b, _c = [float(i) for i in line.split(",")]
        bins.append(_b)
        count.append(_c)

# 가우시안 함수 정의
def gaussian(x, a, b, c):
    return a * np.exp(-((x - b)**2) / (2 * c**2))

# 두 개의 가우시안 함수로 나타내기
def double_gaussian(x, a1, b1, c1, a2, b2, c2):
    return gaussian(x, a1, b1, c1) + gaussian(x, a2, b2, c2)

# 데이터 설정
bins = np.array(bins)
count = np.array(count)

# 피팅 함수에 사용할 초기값 정의 (각 입자의 a, b, c initial 설정)

a1_initial = max(count)
b1_initial = np.mean(bins) - 1
c1_initial = np.std(bins)

a2_initial = max(count) / 2
b2_initial = np.mean(bins) + 1
c2_initial = np.std(bins) / 2

# 피팅 함수 적용
popt, _ = curve_fit(double_gaussian, bins, count, 
                    p0=[a1_initial, b1_initial, c1_initial, a2_initial, b2_initial, c2_initial])

# 피팅 결과 출력
a1, b1, c1, a2, b2, c2 = popt

# 히스토그램 그리기
plt.bar(bins, count, width=0.1, alpha=0.6, color='g', label='Data of two particles')
plt.title('Count of measured energy')
plt.xlabel('Energy')
plt.ylabel('Count')
plt.legend()
plt.grid()
plt.show()

# 피팅된 가우시안 그래프 그리기
x_fit = np.linspace(min(bins), max(bins), 1500)

plt.plot(bins, count, 'yo', label='Data')

# Particle 1의 가우시안
plt.plot(x_fit, gaussian(x_fit, a1, b1, c1), 'c--', label='Particle 1')

# Particle 2의 가우시안
plt.plot(x_fit, gaussian(x_fit, a2, b2, c2), 'm--', label='Particle 2')
plt.title('Gaussian Fitting of two particles')
plt.xlabel('Energy')
plt.ylabel('Count')
plt.legend()
plt.show()

# 각 Particle의 가우시안 그래프의 넓이 적분
area_A = trapezoid(gaussian(x_fit, a1, b1, c1), x=x_fit)
area_B = trapezoid(gaussian(x_fit, a2, b2, c2), x=x_fit)

# 생성비 구하기
ratio = area_A / area_B
print(f"Creation_ratio A:B = {ratio:.2f}")
