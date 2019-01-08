# This link might be useful, unclear yet: https://medium.com/randomai/ensemble-and-store-models-in-keras-2-x-b881a6d7693f
import os
import numpy as np
import keras
from keras import backend
from keras.models import load_model
from collections import defaultdict
import tensorflow as tf
from utils import *

import sys
sys.path.insert(0,'/afs/ece.cmu.edu/usr/sadom/Private/Deep_Learning/Group_Project/11785/cleverhans/')
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import ProjectedGradientDescent, CarliniWagnerL2, FastGradientMethod, BasicIterativeMethod, MomentumIterativeMethod

def test_keras_model(model, SEED, images, labels):
	x_shuffle = permute_pixels(images, SEED)
	pred = np.argmax(model.predict(x_shuffle), axis = 1)
	acc =  np.mean(np.equal(pred, labels))
	return (pred, acc)


def attack_ensemble(DATASET_NAME, ATTACK_NAME,ord,eps,x_test,y_test):
	if ATTACK_NAME == 'fgsm':
		AttackModel = FastGradientMethod
		attack_params = {'eps': eps,'clip_min': 0.0,'clip_max': 1.0,'ord':ord}
	elif ATTACK_NAME == 'pgd':
		AttackModel = ProjectedGradientDescent
		attack_params = {'eps': eps,'clip_min': 0.0,'clip_max': 1.0,'ord':ord}
	elif ATTACK_NAME == 'bim':
		AttackModel = BasicIterativeMethod
		attack_params = {'eps': eps,'clip_min': 0.0,'clip_max': 1.0,'ord':ord}
	elif ATTACK_NAME == 'mim':
		AttackModel = MomentumIterativeMethod
		attack_params = {'eps': eps,'clip_min': 0.0,'clip_max': 1.0,'ord':ord} 
	elif ATTACK_NAME == 'cw':
		AttackModel = CarliniWagnerL2
		attack_params = {'eps': eps,'clip_min': 0.0,'clip_max': 1.0,'ord':ord}
	num_classes = 10
	backend.set_learning_phase(False)
	sess =  backend.get_session()

	"""
	# Define input TF placeholder
	if DATASET_NAME == 'mnist':
		x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
		y = tf.placeholder(tf.float32, shape=(None, 10))
	elif DATASET_NAME == 'cifar10':
		x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
		y = tf.placeholder(tf.float32, shape=(None, 10))
	"""    
	# ## Generate Adversarial Examples
	KNOWN_SEED = 87
	
	# we're going to give this model trained with KNOWN_SEED to the adversary
	known_model = load_model('models/'+DATASET_NAME+'_trained_keras_model'+'.hdf5', custom_objects={'tf':tf}) 
	pred, acc = test_keras_model(known_model, KNOWN_SEED, x_test, y_test)
	print("The normal test accuracy is: {}".format(acc))

	# generate adversariale examples (x_adv) using the known model
	# http://everettsprojects.com/2018/01/30/mnist-adversarial-examples.html
	# https://cleverhans.readthedocs.io/en/latest/source/attacks.html#generate_np
	wrap = KerasModelWrapper(known_model)
	attack_model = AttackModel(wrap, sess=sess)
	x_adv = attack_model.generate_np(x_test, **attack_params)

	# test x_adv against the single model
	pred, acc = test_keras_model(known_model, KNOWN_SEED, x_adv, y_test)
	print("The adversarial test accuracy is: {}".format(acc))

	# ## Attack Ensemble PPD models
	# test x_adv against the ensemble model
	num_models = 10  # debugging with smaller number. change this to 10/50 later.
	num_samples = x_adv.shape[0]
	adv_acc = []
	normal_acc = []
	# We're assuming majority voting?
	# we're going to store votes from each model here
	adv_pred = np.zeros((num_samples, num_classes)) 
	normal_pred = np.zeros((num_samples, num_classes)) 

	for SECRET_SEED in range(num_models):
		keras_model = load_model('models/'+DATASET_NAME+'_trained_keras_model_'+str(SECRET_SEED)+'.hdf5', custom_objects={'tf':tf})

		pred, acc = test_keras_model(keras_model, SECRET_SEED, x_test, y_test)
		normal_pred[range(num_samples), pred] += 1  # +1 vote 
		#print ('SECRET_SEED:', SECRET_SEED, 'Individual model normal accuracy:', acc)
		normal_acc.append(acc)

		pred, acc = test_keras_model(keras_model, SECRET_SEED, x_adv, y_test)
		adv_pred[range(num_samples), pred] += 1  # +1 vote 
		#print ('SECRET_SEED:', SECRET_SEED, 'Individual model adversarial accuracy:', acc)   
		adv_acc.append(acc)  # accuracy per model, not reported in paper

	# for each sample, find out the class with most votes
	normal_pred = np.argmax(normal_pred, axis = 1)
	nor_acc =  np.mean(np.equal(normal_pred, y_test))
	print ('Ensemble normal accuracy:', nor_acc)

	ensemble_pred = np.argmax(adv_pred, axis = 1)
	ens_acc =  np.mean(np.equal(ensemble_pred, y_test))
	print ('Ensemble adversarial accuracy:', ens_acc)
	
	return normal_acc, adv_acc, nor_acc, ens_acc

def get_accuracy():
	attack  = ['fgsm', 'pgd', 'bim', 'mim']
	dataset = ['mnist','cifar10']
	eps_inf = [0.03,0.1,0.2,0.3,0.4]
	eps_mnst= [0.1,0.7,1.1,3.2,4]
	eps_mnst= [x/28.0 for x in eps_mnst]
	eps_cif = [0.3,2,4,6.5,10.5]
	eps_cif = [x/55.42 for x in eps_cif]
	info = defaultdict(dict)

	att = 'mim'
	for data in dataset:
		data_t = get_dataset(data)
		temp = [z.astype('float32') for z in data_t]
		#x_train = temp[0]
		#y_train = temp[1]
		x_test  = temp[2]
		y_test  = temp[3]
		for i in range(2):
			if i == 1:
				ord = 2
				if data == 'mnist':
					eps = eps_mnst
				else:
					eps = eps_cif
			else:
				ord = np.inf
				eps = eps_inf
			for ep in eps:
				print("########Starting ensemble ", att, " on ", data, " with eps=", ep, " and ord=",ord, "#####")
				normal_acc, adv_acc, nor_acc_avg, ens_acc_avg = attack_ensemble(data,att,ord,ep,x_test,y_test)
				info[data][str(ord)+str(ep)] = [normal_acc, adv_acc, nor_acc_avg, ens_acc_avg]
				print("########Ensemble ", att, " on ", data, " with eps=", ep, " and ord=",ord, "saved!#####\n")
	np.save("attack_summary_pgd.npy",info)
	pdb.set_trace()

get_accuracy()
			
 
			
