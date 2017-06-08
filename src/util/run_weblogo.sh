#!/bin/bash

weblogo -F png_print -D array -A protein -s large -t logo-737  <(cut -f 2 ps51657/res/p737.txt | grep -v _ )> tlogo.png
