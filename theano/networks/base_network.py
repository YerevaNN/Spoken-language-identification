import cPickle as pickle


class BaseNetwork:
	
	def say_name(self):
		return "unknown"
	
	
	def save_params(self, file_name, epoch, **kwargs):
		with open(file_name, 'w') as save_file:
			pickle.dump(
				obj = {
					'params' : [x.get_value() for x in self.params],
					'epoch' : epoch, 
				},
				file = save_file,
				protocol = -1
			)
	
	
	def load_state(self, file_name):
		print "==> loading state %s" % file_name
		epoch = 0
		with open(file_name, 'r') as load_file:
			dict = pickle.load(load_file)
			loaded_params = dict['params']
			for (x, y) in zip(self.params, loaded_params):
				x.set_value(y)
			epoch = dict['epoch']
		return epoch


	def get_batches_per_epoch(self, mode):
		if (mode == 'train' or mode == 'predict_on_train'):
			return len(self.train_list_raw) / self.batch_size
		elif (mode == 'test' or mode == 'predict'):
			return len(self.test_list_raw) / self.batch_size
		else:
			raise Exception("unknown mode")
	
	
	def step(self, batch_index, mode):
		
		if (mode == "train"):
			data, answers = self.read_batch(self.train_list_raw, batch_index)
			theano_fn = self.train_fn
		elif (mode == "test" or mode == "predict"):
			data, answers = self.read_batch(self.test_list_raw, batch_index)
			theano_fn = self.test_fn
		elif (mode == "predict_on_train"):
			data, answers = self.read_batch(self.train_list_raw, batch_index)
			theano_fn = self.test_fn
		else:
			raise Exception("unrecognized mode")
		
		ret = theano_fn(data, answers)
		return {"prediction": ret[0],
				"answers": answers,
				"current_loss": ret[1],
				"log": "",
				}