import os,sys
from scipy.stats import multivariate_normal
import scipy.stats as st
import numpy as np
import math



class General:

	global getDistribution
	def getDistribution(x_train, y_train, x_control_train):
		train = []
		for i in range(0, len(y_train)):
			temp1 = np.append(x_train[i], y_train[i])
			temp2 = np.append(temp1, x_control_train[i])
			train.append(temp2)

		mean = np.mean(train, axis=0)
		cov = np.cov(train, rowvar=0)
		# p, s= np.linalg.eig(cov)
		# print(p)
		# l = len(mean)
		# for i in range(0, l):
		# 	for j in range(0, l):
		# 		if i != j and i != l-2 and j != l-2:
		# 			cov[i][j] = 0

		# p, s = np.linalg.eig(cov)
		# for i in range(0, len(p)):
		# 	if p[i] < 0 :
		# 		p[i] = 0
		# #print(p)
		# cov = np.dot(np.dot(s.transpose(), np.diag(p)), s)

		# train2 = []
		# for data in train:
		# 	data2 = [] 
		# 	for j in data : 
		# 		data2.append(j + 0.0001*np.random.randn(2,3))
		# 	train2.append(data2)
		# train2 = np.transpose(train2)

		# KDEpdf = st.gaussian_kde(train2)
		
		#train2 = []
		# for data in train:
		# 	train2.append(np.log(data))
		# #train2 = np.transpose(train2)

		# mean_exp = np.mean(train2, axis=0)
		# cov_exp = np.cov(train2, rowvar=0)

		# prob = {}
		# for i in range(0, len(x_train[0])):
		# 	column = []
		# 	for elem in x_train:
		# 		column.append(elem[i])

		# 	if i == 6:
		# 		mean = np.mean(column)
		# 		std = np.std(column)
		# 		prob[i] = (mean, std)
		# 	else:
		# 		prob[i] = sum(column)/(float(len(column)))

		# tot = 0
		# for j in y_train:
		# 	if j == 1:
		# 		tot += 1
		# prob[i+1] = float(tot)/len(y_train)
		# prob[i+2] = sum(x_control_train)/(float(len(x_control_train)))

		dist_params = {"mean":mean, "cov":cov}

		# min_eig = np.min(np.real(np.linalg.eigvals(cov)))
		# if min_eig < 0:
		# 	cov -= 10*min_eig * np.eye(*cov.shape)

		mean_train = np.mean(x_train, axis=0)
		cov_train = np.cov(x_train, rowvar=0)
		# p1, s1= np.linalg.eig(cov_train)
		# l = len(mean_train)
		# for i in range(0, l):
		# 	for j in range(0, l):
		# 		if i != j:
		# 			cov_train[i][j] = 0

		# p1, s1 = np.linalg.eig(cov_train)
		# for i in range(0, len(p1)):
		# 	if p1[i] < 0 :
		# 		p1[i] = 0
		# #print(p)
		# cov_train = np.dot(np.dot(s1.transpose(), np.diag(p1)), s1)

		# x_train2 = []
		# for data in x_train:
		# 	data2 = [] 
		# 	for j in data : 
		# 		data2.append(np.exp(j))
		# 	x_train2.append(data2)
		#train2 = np.transpose(train2)

		# mean_train_exp = np.mean(x_train2, axis=0)
		# cov_train_exp = np.cov(x_train2, rowvar=0)
		# prob_train = dict(prob)
		# del prob_train[8]
		# del prob_train[9]

		dist_params_train = {"mean":mean_train, "cov":cov_train}
		#print(prob)

		# dist_names = ['gamma', 'beta', 'rayleigh', 'norm', 'pareto']
		# params = {}
		# dist_results = []
		# for dist_name in dist_names:
		# 	dist = getattr(st, dist_name)
		# 	param = dist.fit(train)

		# 	params[dist_name] = param
		# 	print(dist_name, param)
			#Applying the Kolmogorov-Smirnov test
			#D, p = st.kstest(train, dist_name, args=param);
			#dist_results.append((dist_name,p))

		#sel_dist,p = (max(dist_results,key=lambda item:item[1]))

		#print(sel_dist, p, params[sel_dist])
		return dist_params, dist_params_train


	global getProbability
	def getProbability(dist_params, x):
		mean = dist_params["mean"]
		cov = dist_params["cov"]
		# kdepdf = dist_params["kde"]
		# temp = ([x]).transpose()
		# print(kdepdf.evaluate(temp))

		# return kdepdf([x])[0]
		# mean_exp = dist_params["mean_exp"]
		# cov_exp = dist_params["cov_exp"]		

		#print(x, mean, cov)
		return multivariate_normal.pdf(x, mean=mean, cov=cov, allow_singular=1)

		# x2 = []
		# for j in x:
		# 	x2.append(np.log(j))
		# p = multivariate_normal.pdf(x2, mean=mean_exp, cov=cov_exp, allow_singular=1)
		# return p

		# prob = dist_params["multi"]
		# p = 1.0
		# for i in range(0, len(x)):
		# 	if i == 6:
		# 		p = p * st.norm.pdf(x[i], prob[i][0], prob[i][1])
		# 	else :
		# 		if x[i] == 1:
		# 			p = p*prob[i]
		# 		else : 
		# 			p = p * (1-prob[i])
		# #print(x, p)
		# return p


	def getRandomSamples(self, dist_params_train):
		mean = dist_params_train["mean"]
		cov = dist_params_train["cov"]
		#prob = dist_params_train["multi"]

		return multivariate_normal(mean, cov, allow_singular=1).rvs(size=20, random_state=12345)

		# s = []
		# for j in range(0,20):
		# 	# x = multivariate_normal(mean, cov, allow_singular=1).rvs(size=1, random_state=12345)
		# 	# x2 = []
		# 	# for j in x:
		# 	# 	x2.append(np.exp(j))
		# 	# s.append(x2)
		# 	x = []
		# 	for i in range(0, len(prob)):
		# 		if i == 6:
		# 			x.append(np.random.normal(prob[i][0], prob[i][1], 1)[0])
		# 		else:
		# 			x.append(np.random.binomial(1,prob[i]))
		# 	s.append(x)

		# #print(s)
		# return s

	def getExpectedGrad(self, dist_params, params, samples, mu,  z_0, z_1, a, b):
		raise NotImplementedError("Expected gradient function not implemented")
		return []

	def getValueForX(self, dist_params, a,b, params, samples,  z_0, z_1, x, flag):
		raise NotImplementedError("GetValueForX function not implemented")
		return 0

	def getFuncValue(self, dist_params, a,b, params, samples,  z_0, z_1):
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

	def gradientDescent(self, dist_params, a, b, samples, z_0, z_1):
		mu = 0.01
		minVal = 100000000
		size = self.getNumOfParams()

		minParam = [0] * size

		for i in range(1,10):
			params = self.getStartParams(i)
			for k in range(1,50):
				grad = self.getExpectedGrad(dist_params, params, samples, mu, z_0, z_1, a, b)

				for j in range(0, len(params)):
					params[j] = params[j] - 0.01 * grad[j]

				funcVal = self.getFuncValue(dist_params, a,b, params, samples, z_0, z_1)
				if funcVal < minVal:
					minVal = funcVal
					minParam = params

		return minParam

	def processAdultData(self, tau, x_train, y_train, x_control_train, x_test, y_test, x_control_test):
		dist_params, dist_params_train =  getDistribution(x_train, y_train, x_control_train)
		eps = 0.01
		L = math.ceil(tau/eps)
		z_1 = sum(x_control_train)/(float(len(x_control_train)))
		z_0 = 1 - z_1
		#print(z_0, z_1)
		p, q  = [0,0],[0,0]
		paramsOpt = []
		maxAcc = 0
		maxGamma = 0
		f = open("german_sr_gamma_acc", "a")

		span = self.getRange(eps, tau)
		for (a,b) in span:
			acc, gamma = 0, 0
			print("-----",a,b)
			samples = self.getRandomSamples(dist_params_train)

			#try : 
			params = self.gradientDescent(dist_params, a, b, samples, z_0, z_1)
			#params = [0,0,0,0]
			print(params)
			y_res = []

			for x in x_train:
				t = self.getValueForX(dist_params, a,b, params, samples,  z_0, z_1, x, 0)
				if t > 0 :
					y_res.append(1)
				else:
					y_res.append(-1)

			acc = self.getAccuracy(y_train, y_res)
			gamma = self.getGamma(y_train, y_res, x_control_train)
			print("\n", acc, gamma, "\n")
			#except Exception as e:
				#acc, gamma = 0, 0
			#	print(e)

			if maxAcc < acc and gamma > tau - 0.1:
				maxGamma = gamma
				maxAcc = acc
				p = a
				q = b
				paramsOpt = params

		y_test_res = []
		for x in x_test:
				t = self.getValueForX(dist_params, p, q, paramsOpt, samples,  z_0, z_1, x, 0)
				if t > 0 :
					y_test_res.append(1)
				else:
					y_test_res.append(-1)
		f.write(str(tau) + " " + str(self.getGamma(y_test, y_test_res, x_control_test)) + " " + str(self.getAccuracy(y_test, y_test_res)) + "\n")
		return y_test_res

	def test_given_data(self, x_train, y_train, x_control_train, x_test, y_test, x_control_test, sensitive_attrs, tau):
		attr = sensitive_attrs[0]
		x_control_train = x_control_train[attr]
		x_control_test = x_control_test[attr]

		l = len(y_train)


		#print(mean, cov)

		return self.processAdultData(tau, x_train, y_train, x_control_train, x_test, y_test, x_control_test)

	global getData
	def getData():
		x_control_train = []
		x_train = []
		y_train = []
		x_control_test = []
		x_test = []
		y_test = []

		folder = sys.argv[1]
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

		return x_train, y_train, x_control_train, x_control_test, x_test, y_test

	global checkNormalFit
	def checkNormalFit(x_train, y_train, x_control_train):
		train = []
		for i in range(0, len(y_train)):
			temp1 = np.append(x_train[i], y_train[i])
			temp2 = np.append(temp1, x_control_train[i])
			train.append(temp2)

		mean = np.mean(train, axis=0)
		cov = np.cov(train, rowvar=0)
		l = len(mean) - 2
		for i in range(0, l):
			for j in range(0, l):
				if i != j:				
					cov[i][j] = 0

		for i in range(0, len(train[0])):
			data = []
			for elem in train:
				data.append(elem[i])

			def cdf(x):
				return st.norm.cdf(x, mean[i], math.sqrt(cov[i][i]))

			print(st.kstest(data, cdf))

	def test_adult_data(self):
		x_train, y_train, x_control_train, x_control_test, x_test, y_test = getData()
		#checkNormalFit(x_train, y_train, x_control_train)

		for i in range(1,16):
			try : 
				tau = i/20
				print("Tau : ", tau)
				y_res = self.processAdultData(tau, x_train, y_train, x_control_train, x_test, y_test, x_control_test)
				self.getStats(y_test, y_res, x_control_test)
				print("\n")
			except Exception as e:
				print(str(tau) + " failed\n" + str(e))

