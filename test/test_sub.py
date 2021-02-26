import subprocess as sb

input_file = '/home/zhaohoj/Videos/LISA.flv'
output_file = '/home/zhaohoj/Videos/xx.mp4'
out_file = '/home/zhaohoj/Videos/output.mp4'
args = ['ffmpeg', '-i', input_file, '-i', out_file, '-c:v', 'copy', '-c:a', 'aac', output_file, '-y']
p = sb.Popen(args, shell=False)
p.wait()
import os

os.remove(output_file)
print("Done")
