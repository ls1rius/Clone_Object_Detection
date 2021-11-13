import os
st=os.popen("nvidia-smi").read() #str(os.system("nvidia-smi"))

import re

for item in st.split('\n'):
    if 'C' in re.split(r"[ ]+", item):
        #if len(re.split(r"[ ]+", item)[-2])==8:
        print("kill -9 "+re.split(r"[ ]+", item)[2])
        os.system("kill -9 "+re.split(r"[ ]+", item)[2])
            #print("aa",re.split(r"[ ]+", item))

print("aa",st)