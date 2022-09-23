echo "Running on dataset Chameleon"
python train.py --dataset chameleon --wd 0 --dropout 0.1 --lr 0.1 --without-relu
echo "Running on dataset Squirrel"
python train.py --dataset squirrel --wd 5e-4 --dropout 0.1 --lr 0.1 --without-relu
echo "Running on dataset Texas"
python train.py --dataset texas --wd 0 --dropout 0 --lr 0.1 --without-relu
echo "Running on dataset Cornell"
python train.py --dataset cornell --wd 5e-4 --dropout 0.5 --lr 0.01 --without-relu
echo "Running on dataset Wisconsin"
python train.py --dataset wisconsin --wd 5e-4 --dropout 0.5 --lr 0.05 --without-relu
echo "Running on dataset Actor"
python train.py --dataset actor --wd 5e-4 --dropout 0.5 --lr 0.1 --hidden 128 --without-relu