import os,sys
from scipy.stats import multivariate_normal
import scipy.stats as st
import numpy as np
import math
from General import *


class StatisticalRate_FalseDiscovery(General):

	def getExpectedGrad(self, mean, cov, params, samples, mu,  z_0, z_1, a, b):
		u_1, u_2, l_1, l_2, v_1, v_2 = params[0], params[1], params[2], params[3], params[4], params[5]
		a_1, a_2, b_1, b_2 = a[0], a[1], b[0], b[1]
		res1 = []
		res2 = []
		res3 = []
		res4 = []
		res5 = []
		res6 = []
		for x in samples:
				temp = np.append(np.append(x, 1), 1)
				prob_1_1 = multivariate_normal.pdf(temp, mean=mean, cov=cov, allow_singular=1)

				temp = np.append(np.append(x, -1), 1)
				prob_m1_1 = multivariate_normal.pdf(temp, mean=mean, cov=cov, allow_singular=1)

				temp = np.append(np.append(x, 1), 0)
				prob_1_0 = multivariate_normal.pdf(temp, mean=mean, cov=cov, allow_singular=1)

				temp = np.append(np.append(x, -1), 0)
				prob_m1_0 = multivariate_normal.pdf(temp, mean=mean, cov=cov, allow_singular=1)


				prob_y_1 = (prob_1_1 + prob_1_0) / (prob_1_1 + prob_1_0 + prob_m1_0 + prob_m1_1)
				#print(prob_y_1)

				prob_z_0 = (prob_m1_0 + prob_1_0) / (prob_1_1 + prob_1_0 + prob_m1_0 + prob_m1_1)
				prob_z_1 = (prob_m1_1 + prob_1_1) / (prob_1_1 + prob_1_0 + prob_m1_0 + prob_m1_1)


				probc_m1_0 = prob_m1_0 / (prob_1_1 + prob_1_0 + prob_m1_0 + prob_m1_1)
				probc_m1_1 = prob_m1_1 / (prob_1_1 + prob_1_0 + prob_m1_0 + prob_m1_1)

				c_0 = prob_y_1 - 0.5
				c_1 = u_1 * (probc_m1_0 - a_2*prob_z_0) + u_2 * (probc_m1_1 - a_2*prob_z_1)
				c_2 = l_1 * (- probc_m1_0 + b_2*prob_z_0) + l_2 * (- probc_m1_1 + b_2*prob_z_1)
				c_3 = v_1 * prob_z_0/z_0 + v_2 * prob_z_1/z_1

				t = math.sqrt((c_0 + c_1 + c_2 + c_3)*(c_0 + c_1 + c_2 + c_3) + mu*mu)
				t1 = (c_0 + c_1 + c_2 + c_3) * (probc_m1_0 - a_2*prob_z_0)/t
				t2 = (c_0 + c_1 + c_2 + c_3) * (probc_m1_1 - a_2*prob_z_1)/t
				t3 = (c_0 + c_1 + c_2 + c_3) * (- probc_m1_0 + b_2*prob_z_0)/t
				t4 = (c_0 + c_1 + c_2 + c_3) * (- probc_m1_1 + b_2*prob_z_1)/t
				t5 = (c_0 + c_1 + c_2 + c_3) * (prob_z_0/z_0)/t
				t6 = (c_0 + c_1 + c_2 + c_3) * (prob_z_1/z_1)/t

				#print(t1,t2)
				res1.append(t1)
				res2.append(t2)
				res3.append(t3)
				res4.append(t4)
				res5.append(t5)
				res6.append(t6)

		dv1 = np.mean(res5) - b_1 + (b_1-a_1)/2 + (b_1-a_1)* v_1 / (2* math.sqrt(v_1*v_1 + mu*mu)) 
		dv2 = np.mean(res6) - b_1 + (b_1-a_1)/2 + (b_1-a_1)* v_2 / (2* math.sqrt(v_2*v_2 + mu*mu)) 
		return [np.mean(res1), np.mean(res2), np.mean(res3), np.mean(res4), dv1, dv2]

	def getValueForX(self, mean, cov, a,b, params, samples,  z_0, z_1, x):
			u_1, u_2, l_1, l_2, v_1, v_2 = params[0], params[1], params[2], params[3], params[4], params[5]
			a_1, a_2, b_1, b_2 = a[0], a[1], b[0], b[1]

			temp = np.append(np.append(x, 1), 1)
			prob_1_1 = multivariate_normal.pdf(temp, mean=mean, cov=cov, allow_singular=1)

			temp = np.append(np.append(x, -1), 1)
			prob_m1_1 = multivariate_normal.pdf(temp, mean=mean, cov=cov, allow_singular=1)

			temp = np.append(np.append(x, 1), 0)
			prob_1_0 = multivariate_normal.pdf(temp, mean=mean, cov=cov, allow_singular=1)

			temp = np.append(np.append(x, -1), 0)
			prob_m1_0 = multivariate_normal.pdf(temp, mean=mean, cov=cov, allow_singular=1)


			prob_y_1 = (prob_1_1 + prob_1_0) / (prob_1_1 + prob_1_0 + prob_m1_0 + prob_m1_1)
			#print(prob_y_1)

			prob_z_0 = (prob_m1_0 + prob_1_0) / (prob_1_1 + prob_1_0 + prob_m1_0 + prob_m1_1)
			prob_z_1 = (prob_m1_1 + prob_1_1) / (prob_1_1 + prob_1_0 + prob_m1_0 + prob_m1_1)


			probc_m1_0 = prob_m1_0 / (prob_1_1 + prob_1_0 + prob_m1_0 + prob_m1_1)
			probc_m1_1 = prob_m1_1 / (prob_1_1 + prob_1_0 + prob_m1_0 + prob_m1_1)

			c_0 = prob_y_1 - 0.5
			c_1 = u_1 * (probc_m1_0 - a_2*prob_z_0) + u_2 * (probc_m1_1 - a_2*prob_z_1)
			c_2 = l_1 * (- probc_m1_0 + b_2*prob_z_0) + l_2 * (- probc_m1_1 + b_2*prob_z_1)
			c_3 = v_1 * prob_z_0/z_0 + v_2 * prob_z_1/z_1

			t = c_0 + c_1 + c_2 + c_3
			return t

	def getFuncValue(self, mean, cov, a,b, params, samples,  z_0, z_1):
		res = []
		for x in samples:
				t = abs(self.getValueForX(mean, cov, a,b, params, samples,  z_0, z_1, x))
				res.append(t)

		v_1 = params[4]
		v_2 = params[5]
		a_1 = a[0]
		b_1 = b[0]

		exp = np.mean(res)
		result = exp - b_1*v_1 - b_1*v_2
		if v_1 > 0 :
			result += (b_1-a_1)*v_1
		if v_2 > 0 :
			result += (b_1-a_1)*v_2

		return result

	def getNumOfParams(self):
		return 6

	def getStartParams(self, i):
		return [i, i, i, i, -5 + i, -5 + i]

	def getGamma(self, y_test, y_res, x_control_test):
			pos_0 = 0
			pos_1 = 0

			z1_0 = 0
			z1_1 = 0
			for j in range(0,len(y_test)):
				result = y_res[j]

				if result == 1 and x_control_test[j] == 0:
					z1_0 += 1
				if result == 1 and x_control_test[j] == 1:
					z1_1 += 1

				actual = y_test[j]
				if result == 1 and actual == -1 and x_control_test[j] == 0:
					pos_0 += 1
				if result == 1 and actual == -1 and x_control_test[j] == 1:
					pos_1 += 1


			pos_0 = float(pos_0)/z1_0
			pos_1 = float(pos_1)/z1_1
			if pos_0 == 0 or pos_1 == 0:
				return 0
			else:			
				return min(pos_0/pos_1 , pos_1/pos_0)

	def getRange(self, eps, tau):
		span = []
		L = math.ceil(tau/eps)
		for i in range(1, L+1, 10):
			for k in range(1, L+1, 10):		
				a = (i-1) * eps
				b = (i) * eps/ tau
				a_1 = (k-1) * eps
				b_1 = (k) * eps / tau
				if b > 1:
					b = 1.0
				if b_1 > 1:
					b_1 = 1.0

				span.append(([a, a_1],[b, b_1]))
		
		return span

if __name__ == '__main__':
	obj = StatisticalRate_FalseDiscovery()
	obj.test_adult_data()
