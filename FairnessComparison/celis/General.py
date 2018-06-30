import os,sys
from scipy.stats import multivariate_normal
import scipy.stats as st
import numpy as np
import math


class General:

	def getRandomSamples(self, mean, cov):
		return multivariate_normal(mean, cov, allow_singular=1).rvs(size=20, random_state=12345)

	def getExpectedGrad(self, mean, cov, params, samples, mu,  z_0, z_1, a, b):
		raise NotImplementedError("Expected gradient function not implemented")
		return []

	def getValueForX(self, mean, cov, a,b, params, samples,  z_0, z_1, x):
		raise NotImplementedError("GetValueForX function not implemented")
		return 0

	def getFuncValue(self, mean, cov, a,b, params, samples,  z_0, z_1):
		raise NotImplementedError("Value function not implemented")
		return 0

	def getNumOfParams(self):
		raise NotImplementedError("Specify number of params")
		return 0

	def getRange(self, eps, tau):
		span = []
		L = math.ceil(tau/eps)
		for i in range(1, L+1, 10):
			a = (i-1) * eps
			b = (i) * eps / tau
			if b > 1:
				b = 1.0

			span.append(([a, -1],[b, -1]))
		return span

	def getGamma(self, y_test, y_res, x_control_test):
		raise NotImplementedError("Gamma function not implemented")
		return 0

	def getStartParams(self, i):
		num = self.getNumOfParams()
		return [i] * num

	def getAccuracy(self, y_test, y_res):
			total = 0
			fail = 0
			pos_0 = 0
			pos_1 = 0
			for j in range(0,len(y_test)):
				result = y_res[j]
				actual = y_test[j]

				total += 1
				if actual != result:
					fail += 1


			return 1 - fail/(float(total))

	def getStats(self, y_test, y_res, x_control_test):
		try:
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
					print("Observed tau : 0")
			else:			
				print("FDR tau : ", min(pos_0/pos_1 , pos_1/pos_0))

			print("\n")

			total = 0
			fail = 0
			pos_0 = 0
			pos_1 = 0
			for j in range(0,len(y_test)):
				result = y_res[j]
				actual = y_test[j]

				total += 1
				if actual != result:
					fail += 1


			print("Accuracy : ", fail, total, 1 - fail/(float(total))) 

			pos_0 = 0
			pos_1 = 0

			z1_0 = 0
			z1_1 = 0
			for j in range(0,len(y_test)):
				result = y_res[j]
				actual = y_test[j]

				if x_control_test[j] == 0:
					z1_0 += 1
				if x_control_test[j] == 1:
					z1_1 += 1

				if result == 1 and x_control_test[j] == 0:
					pos_0 += 1
				if result == 1 and x_control_test[j] == 1:
					pos_1 += 1


			pos_0 = float(pos_0)/z1_0
			pos_1 = float(pos_1)/z1_1
			if pos_0 == 0 or pos_1 == 0:
					print("Observed tau : 0")
			else:			
				print("SR tau : ", min(pos_0/pos_1 , pos_1/pos_0))


			pos_0 = 0
			pos_1 = 0

			z1_0 = 0
			z1_1 = 0
			for j in range(0,len(y_test)):
				result = y_res[j]
				actual = y_test[j]

				if actual == -1 and x_control_test[j] == 0:
					z1_0 += 1
				if actual == -1 and x_control_test[j] == 1:
					z1_1 += 1

				if result == 1 and actual == -1 and x_control_test[j] == 0:
					pos_0 += 1
				if result == 1 and actual == -1 and x_control_test[j] == 1:
					pos_1 += 1


			pos_0 = float(pos_0)/z1_0
			pos_1 = float(pos_1)/z1_1
			if pos_0 == 0 or pos_1 == 0:
					print("Observed tau : 0")
			else:			
				print("FPR tau : ", min(pos_0/pos_1 , pos_1/pos_0))


			pos_0 = 0
			pos_1 = 0

			z1_0 = 0
			z1_1 = 0
			for j in range(0,len(y_test)):
				result = y_res[j]
				actual = y_test[j]

				if actual == 1 and x_control_test[j] == 0:
					z1_0 += 1
				if actual == 1 and x_control_test[j] == 1:
					z1_1 += 1

				if result == -1 and actual == 1 and x_control_test[j] == 0:
					pos_0 += 1
				if result == -1 and actual == 1 and x_control_test[j] == 1:
					pos_1 += 1


			pos_0 = float(pos_0)/z1_0
			pos_1 = float(pos_1)/z1_1
			if pos_0 == 0 or pos_1 == 0:
					print("Observed tau : 0")
			else:			
				print("FNR tau : ", min(pos_0/pos_1 , pos_1/pos_0))

			pos_0 = 0
			pos_1 = 0

			z1_0 = 0
			z1_1 = 0
			for j in range(0,len(y_test)):
				result = y_res[j]
				actual = y_test[j]

				if actual == 1 and x_control_test[j] == 0:
					z1_0 += 1
				if actual == 1 and x_control_test[j] == 1:
					z1_1 += 1

				if result == 1 and actual == 1 and x_control_test[j] == 0:
					pos_0 += 1
				if result == 1 and actual == 1 and x_control_test[j] == 1:
					pos_1 += 1


			pos_0 = float(pos_0)/z1_0
			pos_1 = float(pos_1)/z1_1
			if pos_0 == 0 or pos_1 == 0:
					print("Observed tau : 0")
			else:			
				print("TPR tau : ", min(pos_0/pos_1 , pos_1/pos_0))


			pos_0 = 0
			pos_1 = 0

			z1_0 = 0
			z1_1 = 0
			for j in range(0,len(y_test)):
				result = y_res[j]
				actual = y_test[j]

				if actual == -1 and x_control_test[j] == 0:
					z1_0 += 1
				if actual == -1 and x_control_test[j] == 1:
					z1_1 += 1

				if result == -1 and actual == -1 and x_control_test[j] == 0:
					pos_0 += 1
				if result == -1 and actual == -1 and x_control_test[j] == 1:
					pos_1 += 1


			pos_0 = float(pos_0)/z1_0
			pos_1 = float(pos_1)/z1_1
			if pos_0 == 0 or pos_1 == 0:
					print("Observed tau : 0")
			else:			
				print("TNR tau : ", min(pos_0/pos_1 , pos_1/pos_0))


			pos_0 = 0
			pos_1 = 0

			z1_0 = 0
			z1_1 = 0
			for j in range(0,len(y_test)):
				result = y_res[j]
				actual = y_test[j]

				if x_control_test[j] == 0:
					z1_0 += 1
				if x_control_test[j] == 1:
					z1_1 += 1

				if result == actual and x_control_test[j] == 0:
					pos_0 += 1
				if result == actual and x_control_test[j] == 1:
					pos_1 += 1


			pos_0 = float(pos_0)/z1_0
			pos_1 = float(pos_1)/z1_1
			if pos_0 == 0 or pos_1 == 0:
					print("Observed tau : 0")
			else:			
				print("AR tau : ", min(pos_0/pos_1 , pos_1/pos_0))


			pos_0 = 0
			pos_1 = 0

			z1_0 = 0
			z1_1 = 0
			for j in range(0,len(y_test)):
				result = y_res[j]

				if result == -1 and x_control_test[j] == 0:
					z1_0 += 1
				if result == -1 and x_control_test[j] == 1:
					z1_1 += 1

				actual = y_test[j]
				if result == -1 and actual == 1 and x_control_test[j] == 0:
					pos_0 += 1
				if result == -1 and actual == 1 and x_control_test[j] == 1:
					pos_1 += 1


			pos_0 = float(pos_0)/z1_0
			pos_1 = float(pos_1)/z1_1
			if pos_0 == 0 or pos_1 == 0:
					print("Observed tau : 0")
			else:			
				print("FOR tau : ", min(pos_0/pos_1 , pos_1/pos_0))

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
				if result == 1 and actual == 1 and x_control_test[j] == 0:
					pos_0 += 1
				if result == 1 and actual == 1 and x_control_test[j] == 1:
					pos_1 += 1


			pos_0 = float(pos_0)/z1_0
			pos_1 = float(pos_1)/z1_1
			if pos_0 == 0 or pos_1 == 0:
					print("Observed tau : 0")
			else:			
				print("PPR tau : ", min(pos_0/pos_1 , pos_1/pos_0))

			pos_0 = 0
			pos_1 = 0

			z1_0 = 0
			z1_1 = 0
			for j in range(0,len(y_test)):
				result = y_res[j]

				if result == -1 and x_control_test[j] == 0:
					z1_0 += 1
				if result == -1 and x_control_test[j] == 1:
					z1_1 += 1

				actual = y_test[j]
				if result == -1 and actual == -1 and x_control_test[j] == 0:
					pos_0 += 1
				if result == -1 and actual == -1 and x_control_test[j] == 1:
					pos_1 += 1


			pos_0 = float(pos_0)/z1_0
			pos_1 = float(pos_1)/z1_1
			if pos_0 == 0 or pos_1 == 0:
					print("Observed tau : 0")
			else:			
				print("NPR tau : ", min(pos_0/pos_1 , pos_1/pos_0))
		except ZeroDivisionError:
			print("Stats inconclusive")

	def gradientDescent(self, mean, cov, a, b, samples, z_0, z_1):
		mu = 0.01
		minVal = 100000000
		size = self.getNumOfParams()

		minParam = [0] * size

		for i in range(1,10):
			params = self.getStartParams(i)
			for k in range(1,50):
				grad = self.getExpectedGrad(mean, cov, params, samples, mu, z_0, z_1, a, b)

				for j in range(0, len(params)):
					params[j] = params[j] - 0.1/k * grad[j]

				funcVal = self.getFuncValue(mean, cov, a,b, params, samples, z_0, z_1)
				if funcVal < minVal:
					minVal = funcVal
					minParam = params

		return minParam

	def processAdultData(self, tau, x_train, y_train, x_control_train, x_test, y_test, x_control_test):
		train = []
		for i in range(0, len(y_train)):
			temp1 = np.append(x_train[i], y_train[i])
			temp2 = np.append(temp1, x_control_train[i])
			train.append(temp2)

		mean = np.mean(train, axis=0)
		cov = np.cov(train, rowvar=0)
		
		# min_eig = np.min(np.real(np.linalg.eigvals(cov)))
		# if min_eig < 0:
		# 	cov -= 10*min_eig * np.eye(*cov.shape)

		mean_train = np.mean(x_train, axis=0)
		cov_train = np.cov(x_train, rowvar=0)

		eps = 0.01
		L = math.ceil(tau/eps)
		z_1 = sum(x_control_train)/(float(len(x_control_train)))
		z_0 = 1 - z_1
		#print(z_0, z_1)
		p, q  = [0,0],[0,0]
		paramsOpt = []
		maxAcc = 0

		span = self.getRange(eps, tau)
		for (a,b) in span:
			print("-----",a,b)

			samples = self.getRandomSamples(mean_train, cov_train)

			params = self.gradientDescent(mean, cov, a, b, samples, z_0, z_1)
			#params = [0,0,0,0]
			print(params)
			y_res = []

			for x in x_train:
				t = self.getValueForX(mean, cov, a,b, params, samples,  z_0, z_1, x)
				if t > 0 :
					y_res.append(1)
				else:
					y_res.append(-1)

			acc = self.getAccuracy(y_train, y_res)
			gamma = self.getGamma(y_train, y_res, x_control_train)
			if maxAcc < acc and gamma >= tau:
				maxAcc = acc
				p = a
				q = b
				paramsOpt = params

		y_test_res = []
		for x in x_test:
				t = self.getValueForX(mean, cov, p, q, paramsOpt, samples,  z_0, z_1, x)

				if t > 0 :
					y_test_res.append(1)
				else:
					y_test_res.append(-1)

		return y_test_res

	def test_given_data(self, x_train, y_train, x_control_train, x_test, y_test, x_control_test, sensitive_attrs, tau):
		attr = sensitive_attrs[0]
		x_control_train = x_control_train[attr]
		x_control_test = x_control_test[attr]

		l = len(y_train)


		#print(mean, cov)

		return self.processAdultData(tau, x_train, y_train, x_control_train, x_test, y_test, x_control_test)

	def test_adult_data(self):

		x_control_train = []
		x_train = []
		y_train = []
		x_control_test = []
		x_test = []
		y_test = []

		folder = "../compass_data"
		temp = []
		with open(folder + "/x_train.txt") as f:
			temp = f.readlines()

		for line in temp:
			temp2 = line[:-1].split(' ')
			a = []
			for i in temp2[:-1]:
				a.append(float(i))
			x_train.append(a)

		temp = []
		with open(folder + "/x_test.txt") as f:
			temp = f.readlines()

		for line in temp:
			temp2 = line.split(' ')
			a = []
			for i in temp2[:-1]:
				a.append(float(i))
			x_test.append(a)

		temp = []
		with open(folder + "/y_train.txt") as f:
			temp = f.readlines()

		for line in temp:
			y_train.append(float(line))

		temp = []
		with open(folder + "/y_test.txt") as f:
			temp = f.readlines()

		for line in temp:
			y_test.append(float(line))

		temp = []
		with open(folder + "/x_control_train.txt") as f:
			temp = f.readlines()

		for line in temp:
			x_control_train.append(float(line))

		temp = []
		with open(folder + "/x_control_test.txt") as f:
			temp = f.readlines()

		for line in temp:
			x_control_test.append(float(line))


		for i in range(9,11):
			tau = i/10
			print("Tau : ", tau)
			y_res = self.processAdultData(tau, x_train, y_train, x_control_train, x_test, y_test, x_control_test)
			self.getStats(y_test, y_res, x_control_test)
			print("\n")