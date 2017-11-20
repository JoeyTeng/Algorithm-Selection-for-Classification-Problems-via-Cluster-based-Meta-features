# @Author: Joey Teng <Toujour>
# @Date:   12-Sep-2017
# @Email:  joey.teng.dev@gmail.com
# @Filename: experiment1.py
# @Last modified by:   Toujour
# @Last modified time: 12-Sep-2017

import re
import sys
import shlex
import subprocess

if __name__ == "__main__":
    print("Start")
    count = 0

    for file in open(sys.argv[1], 'r'):
        print("# Task: {0}\n  File: {1}".format(
            count, re.search(r"/.*?'", file).group(0)[1:-1]))
        count += 1
        Popen = subprocess.Popen(
            "/usr/local/bin/python3 {0} {1}".format(sys.argv[2], file), shell=True)
        try:
            while (not Popen.poll()):
                print("test")
                subprocess.call("/bin/sleep 1", shell=True)
        except KeyboardInterrupt:
            Popen.terminate()
            print("Cleaning")
            subprocess.call("/bin/sleep 10", shell=True)
            raise KeyboardInterrupt
