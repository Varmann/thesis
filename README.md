python predict.py --input /path/to/file.png --output /path/to/file_out.png --model /pathh/to/model.pth


python C:\Users\vmanukyan\Documents\dev\thesis\nets\Nets\Unet\github\predict.py --input C:\Users\vmanukyan\Documents\dev\thesis\nets\Nets\Unet\image_2_1.png --output C:\Users\vmanukyan\Documents\dev\thesis\nets\Nets\Unet\image_2_1_out.png --model C:\Users\vmanukyan\Documents\dev\thesis\nets\Nets\Unet\github\checkpoints\checkpoint.pth --viz


----------------------------------------------------------------------------------------------------------------

https://docs.python.org/3/library/argparse.html#core-functionality


https://stackoverflow.com/questions/52132076/argparse-action-or-type-for-comma-separated-list

names: List[str] = ['Jane', 'Dave', 'John']

parser = argparse.ArumentParser()
parser.add_argument('--names', default=names, action=SplitArgs)

args = parser.parse_args()s
names = args.names




-------------------------------------------
-------------------------------------------
python C:\Users\vmanukyan\Documents\dev\thesis\nets\Nets\Unet\github\predict.py --input C:\Users\vmanukyan\Documents\dev\thesis\nets\Nets\Unet\image_2_1.png --output C:\Users\vmanukyan\Documents\dev\thesis\nets\Nets\Unet\image_2_1_out.png --model C:\Users\vmanukyan\Documents\dev\thesis\nets\Nets\Unet\github\checkpoints\checkpoint_epoch5.pth --viz



--------------------------------------------------------
python C:\Users\vmanukyan\Documents\dev\thesis\nets\Nets\Unet\github\predict.py --viz


######### bd9d82b1fec4714a2a9f4cb1217e7a608dfa9787  ApI