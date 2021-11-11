python tools/demo.py image -n yolox-s -c ./yolox_s.pth --path assets/dog.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device gpu



python tools/train.py -n yolox-s -d 4 -b 64 --fp16 -o --cache



python tools/demo.py image -n yolox-s -c ./YOLOX_outputs/yolox_s/latest_ckpt.pth --path assets/1633140199_7534342.jpg --conf 0.5 --nms 0.45 --tsize 640 --save_result --device gpu




python tools/demo.py image -n yolox-s -c ./YOLOX_outputs/yolox_s/latest_ckpt.pth --path ../data/img_clip_all --conf 0.5 --nms 0.45 --tsize 640 --save_result --device gpu



CUDA_VISIBLE_DEVICES=0,1,2,3



