keras:
	python main.py --batch_size 128 --num_classes 10 --epochs 5 --exp_name "Keras"
rfc:
	python main.py --n_estimators 50 --exp_name "RFC"
