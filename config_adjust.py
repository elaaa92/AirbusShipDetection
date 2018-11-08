from os import listdir, getcwd
import re

#Adjust config
conf=''
with open('./models/model/pipeline.config',"r+") as f:
	conf = f.read()
	label_path = re.findall('PATH_TO_BE_CONFIGURED/\w*\.pbtxt',conf)
	for label in label_path:
		conf=conf.replace(label,getcwd()+'/data/label_map.pbtxt')
	label_path = re.findall('label_map_path:\s.+',conf)
	for label in label_path:
		conf=conf.replace(label,'label_map_path: \"'+getcwd()+'/data/label_map.pbtxt\"')
	record_paths = re.findall('PATH_TO_BE_CONFIGURED/\w*\.record',conf)
	if record_paths:
		conf=conf.replace(record_paths[0],getcwd()+'/data/train.record')
		conf=conf.replace(record_paths[1],getcwd()+'/data/eval.record')
	record_paths = re.findall('input_path:\s.+',conf)
	if record_paths:
		conf=conf.replace(record_paths[0],'input_path: \"'+getcwd()+'/data/train.record\"')
		conf=conf.replace(record_paths[1],'input_path: \"'+getcwd()+'/data/eval.record\"')
	saved_model_path = re.findall('PATH_TO_BE_CONFIGURED/\w*\.ckpt',conf)
	if saved_model_path:
		conf=conf.replace(saved_model_path[0],getcwd()+'/models/model/model.ckpt')
	saved_model_path = re.findall('fine_tune_checkpoint:\s.+',conf)
	if saved_model_path:
        	conf=conf.replace(saved_model_path[0],'fine_tune_checkpoint: \"'+getcwd()+'/models/model/model.ckpt\"')
	num_classes = re.findall('num_classes:\s\d*',conf)
	conf=conf.replace(num_classes[0],'num_classes: 1')
	num_steps = re.findall('num_steps:\s\d*',conf)
	conf=conf.replace(num_steps[0],'num_steps: 200000')
	eval_config = re.findall('num_examples:\s\d*',conf)
	conf=conf.replace(eval_config[0],'num_examples: '+str(2000))
	conf=conf.replace('}\ntrain_input_reader','  batch_queue_capacity: 2\n  prefetch_queue_capacity: 2\n}\ntrain_input_reader')
	sample = re.findall('sample_1_of_n_examples:\s1\n',conf)
	if sample:
		conf=conf.replace(sample[0],'')
	f.seek(0)
	f.write(conf)

