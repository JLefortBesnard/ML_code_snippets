import numpy as np
import matplotlib.pylab as plt

class gradient_descent:
	def __init__(self, m):
		self.X = np.squeeze(np.random.rand(1, m)*np.random.choice([-1, 1], size=m) + np.random.randint(0, 5, size=m))
		self.m = m
		self.theta0_truth = np.random.randint(-5, 5)
		self.theta1_truth = np.random.randint(-5, 5)
		self.noise = [np.random.rand() * np.random.choice([-1, 1]) for i in range(0, self.m)]
		self.Y = self.theta0_truth + self.theta1_truth*self.X + self.noise 

	def display_sample(self):
		print('')
		print('---')
		print('** sample : **')
		print('X : ', self.X)
		print('Y : ', np.round(self.Y, 1))
		print('---')
		print('Underlying truth => Y = ', self.theta0_truth, ' + ', self.theta1_truth, ' * X + noise')
		print('Sample size = ', self.m)
		print('---')
		print('')

	def show_sample_on_graph(self, theta0, theta1, display_new=0):
		plt.scatter(self.X, self.Y, marker='x', c='blue')
		plt.plot([0, 5], [theta0 + theta1*0, theta0 + theta1*5], color='blue')
		if display_new == 1:
			plt.plot([0, 5], [self.new_theta0 + self.new_theta1*0, self.new_theta0 + self.new_theta1*5], color='red')
		plt.xlabel("X")
		plt.xlabel("Y")
		plt.xlim([0, 5])
		plt.ylim([-25, 25])
		plt.show()

	def make_prediction(self, theta0, theta1):
		return [theta0 + xi*theta1 for xi in self.X]

	def compute_cost(self, theta0, theta1):
		squared_distance = 0
		for y_pred, yi in zip(self.make_prediction(theta0, theta1), self.Y):
			squared_distance += (y_pred - yi)**2
		cost_value = 1/(2*self.m) * squared_distance
		return cost_value

	def derivative0(self, theta0, theta1, alpha):
		distance = 0
		for y_pred, yi in zip(self.make_prediction(theta0, theta1), self.Y):
			distance += (y_pred - yi)
		der0 = 1/self.m * alpha * distance 
		return der0

	def derivative1(self, theta0, theta1, alpha):
		distance = 0
		for y_pred, yi, xi in zip(self.make_prediction(theta0, theta1), self.Y, self.X):
			distance += (y_pred - yi)*xi
		der1 = 1/self.m * alpha * distance 
		return der1

	def compute_gradient(self, theta0, theta1, alpha):
		self.new_theta0 = theta0 - self.derivative0(theta0, theta1, alpha)
		self.new_theta1 = theta1 - self.derivative1(theta0, theta1, alpha)
		return self.new_theta0, self.new_theta1

theta0 = 0
theta1 = 1
alpha = 0.1

trial1 = gradient_descent(15)
trial1.display_sample()
trial1.show_sample_on_graph(theta0, theta1)
trial1.compute_cost(theta0, theta1)
trial1.compute_gradient(theta0, theta1, alpha)
trial1.show_sample_on_graph(theta0, theta1, display_new=1)
print(trial1.new_theta0, '  ', trial1.new_theta1, '  | cost = ', trial1.compute_cost(trial1.new_theta0, trial1.new_theta1))
trial1.compute_gradient(trial1.new_theta0, trial1.new_theta1, alpha)
print(trial1.new_theta0, '  ', trial1.new_theta1, '  | cost = ', trial1.compute_cost(trial1.new_theta0, trial1.new_theta1))

def run(m, theta0, theta1, alpha, iteration=10):
	LR = gradient_descent(10)
	LR.display_sample()
	LR.show_sample_on_graph(theta0, theta1)
	new_theta0, new_theta1 = LR.compute_gradient(theta0, theta1, alpha)
	LR.show_sample_on_graph(theta0, theta1, display_new=1)
	print(LR.new_theta0, '  ', LR.new_theta1, '  | cost = ', LR.compute_cost(LR.new_theta0, LR.new_theta1))
	for i in range(iteration):
		print('--')
		old_theta0, old_theta1 = new_theta0, new_theta1
		print(old_theta0, old_theta1)
		new_theta0, new_theta1 = LR.compute_gradient(old_theta0, old_theta1, alpha)
		print(new_theta0, new_theta1)
		LR.show_sample_on_graph(old_theta0, old_theta1, display_new=1)
		print(LR.new_theta0, '  ', LR.new_theta1, '  | cost = ', LR.compute_cost(LR.new_theta0, LR.new_theta1))
run(100, 0, 1, 0.00001)
run(100, 0, 1, 1)
run(10, 0, 1, 0.01)


def automatic_fitting(m, theta0, theta1, alpha):
	LR = gradient_descent(m)
	LR.display_sample()
	LR.show_sample_on_graph(theta0, theta1)
	LR.compute_gradient(theta0, theta1, alpha)
	print(LR.new_theta0, '  ', LR.new_theta1, '  | cost = ', LR.compute_cost(LR.new_theta0, LR.new_theta1))
	while LR.compute_cost(LR.new_theta0, LR.new_theta1) > 0.3:
		LR.compute_gradient(LR.new_theta0, LR.new_theta1, alpha)
		print(LR.new_theta0, '  ', LR.new_theta1, '  | cost = ', LR.compute_cost(LR.new_theta0, LR.new_theta1))
	print('Reach a minimum')
	LR.show_sample_on_graph(theta0, theta1, display_new=1)


automatic_fitting(10, 0, 1, 0.1)