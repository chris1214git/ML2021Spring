# python hw11_domain_adaptation.py --epochs 200 --Canny_low 170 --lamb_decay 10 --log_dir DaNN1-1 --cuda 3
# python hw11_domain_adaptation.py --epochs 200 --Canny_low 200 --lamb_decay 10 --log_dir DaNN1-2 --cuda 3
# python hw11_domain_adaptation.py --epochs 200 --Canny_low 270 --lamb_decay 10 --log_dir DaNN1-3 --cuda 3 
# python hw11_domain_adaptation.py --epochs 200 --Canny_low 270 --lamb_decay 5 --log_dir DaNN1-3-2 --cuda 3 

python hw11_domain_adaptation.py --epochs 1000 --Canny_low 270 --lamb_decay 10 --log_dir DaNN2-1-1 --cuda 3
python hw11_domain_adaptation.py --epochs 2000 --Canny_low 270 --lamb_decay 10 --log_dir DaNN2-1-2 --cuda 3
python hw11_domain_adaptation.py --epochs 4000 --Canny_low 270 --lamb_decay 10 --log_dir DaNN2-1-3 --cuda 3

# python hw11_domain_adaptation.py --epochs 1000 --Canny_low 270 --lamb_decay 10 --log_dir DaNN2-1-1 --cuda 3
# python hw11_domain_adaptation.py --epochs 1000 --Canny_low 270 --lamb_decay 10 --log_dir DaNN2-1-1 --cuda 3
# python hw11_domain_adaptation.py --epochs 1000 --Canny_low 270 --lamb_decay 10 --log_dir DaNN2-1-1 --cuda 3
